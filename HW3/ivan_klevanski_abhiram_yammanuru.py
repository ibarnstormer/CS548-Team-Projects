"""
CS 548 Team Project 3

Authors: Ivan Klevanski Abhiram Yammanuru


Notes:

Place all files (csvs) in the same
directory as the source file

"""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import copy
import os
import random
import traceback
import time
import sklearn.metrics as skl_m
import sklearn.model_selection as skl_ms
import torch.nn as nn
import torch.cuda
import torch.utils
import torch.utils.data
import torchvision
import albumentations
import albumentations.pytorch
import torch.nn.functional as F
import warnings

from torch.utils.data import Dataset
from timm import utils
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from timm.models import vgg
from timm.models import resnet
from timm.models import vision_transformer
from timm.models import fastvit
from lime import lime_image

warnings.filterwarnings("ignore")

abs_path = os.path.dirname(os.path.abspath(__file__))

plot_eda_graphs = False

# Params

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
seed = 42
image_resize = 48

num_classes = 1000 # Models from timm use 1000 as default number of classes

""" ------ Core Classes ------ """

class ImageDataSet(Dataset):
    """
    PyTorch Dataset for image data
    """

    oversample_preprocessing = albumentations.Compose([
            albumentations.VerticalFlip(),
            albumentations.pytorch.ToTensorV2()
    ])

    def __init__(self, images: list, oversamples: list, labels: list):
        self.images = np.asarray(images) / 255
        self.oversamples = oversamples
        self.labels = labels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, item):

        image = np.asarray(self.images[item]).reshape(1, image_resize, image_resize)
        oversample = self.oversamples[item]

        if oversample == 1:
            image = self.oversample_preprocessing(image=image)["image"].permute(1, 2, 0)
        else:
            image = torch.from_numpy(image)

        # Format/adjust image
        label = self.labels[item]
        return image, label


class MLPRegression(nn.Module):
    """
    Multi-Layer Perceptron regression model
    """
    def __init__(self, dim, drop_rate: float = 0):
        super(MLPRegression, self).__init__()
        self.mlp = torchvision.ops.MLP(in_channels=dim**2, dropout=drop_rate, hidden_channels=[int((dim**2) * 2), int((dim**2) * 3), int((dim**2) * 5), int((dim**2) * 3), int((dim**2) * 2), num_classes])   
        self.linear = nn.Linear(in_features=num_classes, out_features=1)

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = self.mlp(x)
        x = self.linear(x)
        return x


class CNNetRegression(nn.Module):
    """
    Generic Convolutional Neural Network for image regression task.<br>
    Note: dim assumes that image is square
    """

    def __init__(self, dim, color_dim, drop_rate=0, use_batch_norm=True):
        super(CNNetRegression, self).__init__()

        self.use_batch_norm = use_batch_norm

        fc1_dim = int(((dim - 4) / 2) / 2) # dim - (conv1 kernel_size / conv1 stride) / (maxpool2 stride)

        self.conv1 = nn.Conv2d(in_channels=color_dim, out_channels=48, kernel_size=(4, 4), stride=2)
        self.conv2 = nn.Conv2d(in_channels=48, out_channels=96, kernel_size=(4, 4), stride=1, padding=2)
        self.conv3 = nn.Conv2d(in_channels=96, out_channels=192, kernel_size=(3, 3), stride=1)

        self.mp1 = nn.MaxPool2d(kernel_size=(2, 2), stride=1, padding=1)
        self.mp2 = nn.MaxPool2d(kernel_size=(2, 2), stride=2)
        
        self.bn1 = nn.BatchNorm2d(48)
        self.bn2 = nn.BatchNorm2d(96)
        self.bn3 = nn.BatchNorm2d(192)

        self.fc1 = nn.Linear(in_features=(fc1_dim ** 2) * 192, out_features=8192)
        self.fc2 = nn.Linear(in_features=8192, out_features=4096)
        self.fc3 = nn.Linear(in_features=4096, out_features=num_classes) # num_classes = 1000
        self.linear = nn.Linear(in_features=num_classes, out_features=1)

        self.dropout = nn.Dropout(p=drop_rate)

    def forward(self, x):
        x = self.conv1(x)
        if self.use_batch_norm:
            x = self.bn1(x)
        x = F.relu(x)
        x = self.mp1(x)
        x = self.dropout(x)

        x = self.conv2(x)
        if self.use_batch_norm:
            x = self.bn2(x)
        x = F.relu(x)
        x = self.mp2(x)
        x = self.dropout(x)

        x = self.conv3(x)
        if self.use_batch_norm:
            x = self.bn3(x)
        x = F.relu(x)
        x = self.mp1(x)
        x = self.dropout(x)

        x = torch.flatten(x, 1)

        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)

        x = self.fc2(x)
        x = F.relu(x)
        x = self.dropout(x)

        x = self.fc3(x)
        x = F.relu(x)
        x = self.dropout(x)

        x = self.linear(x)

        return x


class RegressionModel(nn.Module):
    """
    Wrapper class that converts timm image classifier models<br>
    to a regression model.
    """
    
    def __init__(self, base_model):
        super(RegressionModel, self).__init__()
        self.base = base_model
        self.linear = nn.Linear(in_features=num_classes, out_features=1)
    
    def forward(self, x):
        x = self.base(x)
        x = self.linear(x)

        return x

""" ------ Core Methods ------ """

def setup():
    """
    Set up the application environment
    """
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = True
        torch.cuda.manual_seed(seed)
        cuda_info = "Cuda modules loaded."
    else:
        cuda_info = "Cuda modules not loaded."

    print("[Info]: " + cuda_info + '\n')

    random.seed(seed)
    np.random.seed(seed)
    utils.random_seed(seed, 0)
    torch.manual_seed(seed)


def load_data():
    """
    Loads and initializes all datasets<br>
    **Returns**: dataframe and datasets for regression task
    """

    print("[Info]: Loading data. \n")
    
    df = pd.read_csv(os.path.join(abs_path, "age_gender.csv"))

    # Oversampling for vertical image flip
    df["oversample"] = 0
    oversample_df = df.copy()

    oversample_df["oversample"] = 1
    df = pd.concat([df, oversample_df], ignore_index=True)
    
    # Format pixels and age to float32 and shuffle dataframe
    df["pixels"] = df["pixels"].apply(lambda img: np.array(img.split(' '), dtype=np.float32))
    df["age"] = df["age"].astype(np.float32)
    df = df.sample(frac=1).reset_index(drop=True)

    # Train, Validation, and Test datasets
    train_df, test_df = skl_ms.train_test_split(df, test_size=0.2)
    train_df, validation_df = skl_ms.train_test_split(train_df, test_size=0.15)

    train_set = ImageDataSet(train_df["pixels"].tolist(), train_df["oversample"].tolist(), train_df["age"].tolist())
    validation_set = ImageDataSet(validation_df["pixels"].tolist(), validation_df["oversample"].tolist(), validation_df["age"].tolist())
    test_set = ImageDataSet(test_df["pixels"].tolist(), test_df["oversample"].tolist(), test_df["age"].tolist())

    return df, train_set, validation_set, test_set


def train_model(
    model: nn.Module, 
    train_loader: DataLoader, 
    validation_loader: DataLoader,
    loss_fn = nn.MSELoss(),
    optim: torch.optim.Optimizer = torch.optim.Adam,
    lr: float = 1e-4,
    num_epochs: int = 15,
    specifier = "",
    loss_str = "MSE Loss",
    optim_str = "Adam"):
    """
    Trains the model via backpropagation

    **Returns**: Trained model weights, train losses, validation losses
    """
    
    # Utility methods

    def train(data_loader: DataLoader):
        """
        Single model training step
        """

        model.train()

        run_loss = 0

        # Mini-batch executions
        for images, labels in tqdm(data_loader):
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            # Forward and backward pass to update model
            with torch.set_grad_enabled(True):
                outputs = model(images)
                loss = loss_fn(outputs.squeeze().float(), labels.float())

                loss.backward()
                optimizer.step()

            run_loss += loss.item()

        return run_loss / len(data_loader)

    def validate(data_loader: DataLoader):
        """
        Single model validation step
        """

        model.eval()

        run_loss = 0

        # Mini-batch executions
        for images, labels in tqdm(data_loader):
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            # Forward propagation to get predicted target variable
            with torch.no_grad():
                outputs = model(images)
                loss = loss_fn(outputs.squeeze().float(), labels.float())

            run_loss += loss.item()

        return run_loss / len(data_loader)

    if model is not None:
        try:
            print('-' * 20)
            print("Training: " + specifier)
            print('-' * 20 + '\n')

            # Initialization
            optimizer=optim(model.parameters(), lr=lr)

            m_best_weights = copy.deepcopy(model.state_dict())
            m_best_loss = np.inf
            train_losses = list()
            validation_losses = list()

            model = model.to(device)

            # Epochs
            for epoch in range(num_epochs):

                print('-' * 20)
                print("Epoch: " + str(epoch + 1) + " out of " + str(num_epochs))
                print('-' * 20)

                epoch_loss = train(train_loader)
                print("\nEpoch Summary: Train Loss: {:.4f}".format(epoch_loss))
                train_losses.append(epoch_loss)

                epoch_loss = validate(validation_loader)
                print("\nEpoch Summary: Evaluation Loss: {:.4f}\n".format(epoch_loss))
                validation_losses.append(epoch_loss)

                # Store best model weights based on lowest loss
                if epoch_loss <= m_best_loss:
                    m_best_loss = epoch_loss
                    m_best_weights = copy.deepcopy(model.state_dict())

            print("Lowest validation loss: {:.4f}".format(m_best_loss))
            print("Parameters used: Learning Rate: {:.4f}, Number of Epochs: {}, Loss Function: {}, Optimizer: {}\n".format(lr, num_epochs, loss_str, optim_str))
            return m_best_weights, train_losses, validation_losses
        except Exception:
            print("[Error]: %s training failed due to an exception, exiting...\n" % specifier)
            print("[Error]: Exception occurred during training")
            traceback.print_exc()
            exit(1)


def test_model(
    model: nn.Module, 
    test_loader: DataLoader, 
    specifier = ""):

    if model is not None:
        try:

            print('-' * 20)
            print("Testing: " + specifier)
            print('-' * 20)

            model.eval()

            predictions = []
            ground_truths = []

            # Mini-batch executions
            with torch.no_grad():
                for images, labels in tqdm(test_loader):
                    images = images.to(device)

                    outputs = model(images)

                    # If using batches, append output array to predictions instead of getting first item
                    if outputs.size() == 1:
                        predictions.append(outputs.item())
                    else:
                        predictions = predictions + outputs.cpu().detach().numpy().tolist()

                    ground_truths.append(labels)

                predictions = np.array(predictions)
                ground_truths = np.array(torch.cat(ground_truths))

            print("\n%s MSE: %.4f\n" % (specifier, skl_m.root_mean_squared_error(ground_truths, predictions) ** 2))
            print("\n%s RMSE: %.4f\n" % (specifier, skl_m.root_mean_squared_error(ground_truths, predictions)))
            print("\n%s MAE: %.4f\n" % (specifier, skl_m.mean_absolute_error(ground_truths, predictions)))
            print("\n%s R-Squared: %.4f\n" % (specifier, skl_m.r2_score(ground_truths, predictions)))

        except:
            print("[Error]: Exception occurred during testing:\n")
            traceback.print_exc()
    pass


def model_train_test(
        model_name: str,
        model: nn.Module,
        weights_file_name: str,
        train_loader: DataLoader, 
        validation_loader: DataLoader,
        test_loader: DataLoader,
        losses_df: pd.DataFrame, 
        optimizer: torch.optim.Optimizer = torch.optim.Adam, 
        loss_fn: nn.Module = nn.MSELoss(),
        optim_name: str = "Adam",
        loss_name: str = "MSE Loss",
        lr: float = 1e-4,
        num_epochs: int = 15
        ):
    """
    Trains, tests, and saves the weights for a given model

    **Returns**
    """
    curr_time = time.time()
    best_weights, train_losses, validation_losses = train_model(model=model, 
                                                                train_loader=train_loader, 
                                                                validation_loader=validation_loader, 
                                                                specifier=model_name, 
                                                                optim=optimizer, 
                                                                loss_fn=loss_fn,
                                                                lr=lr,
                                                                num_epochs=num_epochs,
                                                                optim_str=optim_name,
                                                                loss_str=loss_name)
    stop_time = time.time()
    diff = stop_time - curr_time

    print("Total training time for {}: {:.3f} seconds.\n".format(model_name, diff))

    torch.save(best_weights, os.path.join(abs_path, weights_file_name))
    if losses_df is not None:
        losses_df = pd.concat([losses_df, pd.DataFrame({"Model_Name": model_name, 
                                                    "Training_Losses": np.asarray(train_losses),
                                                    "Validation_Losses": np.asarray(validation_losses)})], ignore_index=True)

    model.load_state_dict(best_weights)

    test_model(model=model, test_loader=test_loader, specifier=model_name)

    return losses_df


def model_load_test(
        model_name: str,
        model: nn.Module,
        weights_file_name: str,
        test_loader: DataLoader):
    """
    Loads model weights and tests model
    """

    weights = torch.load(os.path.join(abs_path, weights_file_name), map_location=device)

    model = model.to(device)

    model.load_state_dict(weights)
    test_model(model=model, test_loader=test_loader, specifier=model_name)

""" ------ PyTorch Utility methods ------ """

def visualize_tensor(t: torch.Tensor, plot_title: str = ""):
    """
    Plots a given 4D tensor (collection/batch of 3D tensors (2D tensor plus color dimension)) in a grid-like pattern<br>

    Tensor dimensions: (b, c, h, w)<br>
    **b**: Batch index (single instance of an image-like tensor)<br>
    **c**: Color dimension (1 for grayscale, 3 for colored)<br> 
    **h**: Height dimension<br>
    **w**: Width dimension
    """

    grid = torchvision.utils.make_grid(t, nrow=int(np.ceil(t.shape[0] ** 0.5)))

    plt.imshow(grid.permute(1, 2, 0))
    if plot_title != "":
        plt.title(plot_title)
    plt.show()


def graph_losses(df: pd.DataFrame):
    """
    Graphs training and validation losses
    **df**: DataFrame containing losses per model
    """

    for model_name in df["Model_Name"].unique():
        train_losses = df.loc[df["Model_Name"] == model_name]["Training_Losses"].to_numpy()
        validation_losses = df.loc[df["Model_Name"] == model_name]["Validation_Losses"].to_numpy()

        plt.plot(np.arange(0, len(train_losses), 1), train_losses, label="Training Loss")
        plt.plot(np.arange(0, len(validation_losses), 1), validation_losses, label="Validation Loss")
        plt.title("Training and Validation Losses for {}".format(model_name))
        plt.legend()
        plt.show()


def visualize_image(ds: ImageDataSet, batch_size: int = 16, idx: int = 0):
    """
    Visualizes a mini-batch of images from the dataset used by the neural networks

    **ds**: ImageDataSet object<br>
    **batch_size**: Equivalent to how much images to display<br>
    **idx**: Index for dataloader<br>
    """

    dl = DataLoader(ds, batch_size=batch_size)
    it = iter(dl)
    i = -1

    while i < idx:
        image_batch, _ = next(it)
        i += 1
        if i >= idx:
            visualize_tensor(image_batch)
            break


""" ------ Project Tasks ------ """

def EDA(formattedDF: pd.DataFrame, ds: ImageDataSet):
    """
    Performs Exploratory Data Analysis (EDA) on the image dataset.
    """

    # Loading the Data
    imageData = pd.read_csv(os.path.join(abs_path, "age_gender.csv"))

    # Getting the shape of the data
    print("DataFrame Head: {}".format(imageData.head()))
    print("Shape: {}\n".format(imageData.shape))

    if plot_eda_graphs:

        # Plotting the distribution of Age 
        vcs = imageData['age'].value_counts()
        vcs.plot(kind="bar")
        for idx in vcs.index:
            plt.text(idx, vcs[idx], vcs[idx], ha="center", va="center")
        plt.show()

        # Plotting the distribution of Gender 
        vcs1 = imageData['gender'].value_counts()
        vcs1.plot(kind="bar")
        for idx in vcs1.index:
            plt.text(idx, vcs1[idx], vcs1[idx], ha="center", va="center")
        plt.show()

        # Plotting the distribution of Ethnicity 
        vcs2 = imageData['ethnicity'].value_counts()
        vcs2.plot(kind="bar")
        for idx in vcs2.index:
            plt.text(idx, vcs2[idx], vcs2[idx], ha="center", va="center")

        # Seeing the correlation between Age and Gender 
        sns.scatterplot(x=imageData["age"], y=imageData["gender"])
        plt.show()

        # Seeing the correlation between Gender and Ethnicity 
        sns.scatterplot(x=imageData["gender"], y=imageData["ethnicity"])
        plt.show()
            
        # Seeing the corrlation between Age and Ethnicity
        sns.scatterplot(x=imageData["age"], y=imageData["ethnicity"])
        plt.show()

        # Example of image mini-batch
        visualize_image(ds)


def data_preprocessing():
    # TODO: note: may not be necessary since most image augmentations will happen in dataset class
    pass


def explainability(ds: ImageDataSet):
    """
    Provides an interpretation on how the image regressors make their predictions

    """

    # Load CNN to use as example
    weights = torch.load(os.path.join(abs_path, "cnn.pth"), map_location=torch.device("cpu"))
    model = CNNetRegression(image_resize, 1, use_batch_norm=True)

    model = model.to(torch.device("cpu"))

    model.load_state_dict(weights)
    
    # Visualization of convolution filters (note: just conv1 since subsequent have a color dimension exceeding 3)
    conv1_kernels = model.conv1.weight.data

    visualize_tensor(conv1_kernels, "Visualization of kernels for layer: conv1")


    # LIME Implementation


    pass


def main():

    setup()

    df, train_set, validation_set, test_set = load_data()
    
    # EDA
    EDA(df, train_set)

    losses_df = pd.DataFrame(columns=["Model_Name", "Training_Losses", "Validation_Losses"])

    # Constants
    batch_size = 32
    lr = 1e-4
    num_epochs = 15 # 15 | 25 -> set with dropout 0 | 0.15

    # Hyperparameters for optimization testing
    optimizer = torch.optim.Adam # adam | sgd
    optim_name = "Adam"
    loss_fn = nn.MSELoss() # nn.MSELoss() | nn.L1Loss()
    loss_name = "MSE"
    drop_rate = 0 # 0 | 0.15

    train_loader = DataLoader(train_set, shuffle=True, batch_size=batch_size)
    validation_loader = DataLoader(validation_set, shuffle=True, batch_size=batch_size)
    test_loader = DataLoader(test_set, shuffle=True, batch_size=batch_size)

    do_explain = False # Run explainability analysis
    test_only = False # Whether or not to just run model tests or to train models as well
    plot_losses = True
    save_losses = True
    
    load_mlp = True
    load_cnn = True
    load_vgg = True
    load_resnet = True
    load_vit = True
    load_fastvit = True

    # Model Training and evaluation

    if not test_only:
        # MLP
        if load_mlp:
            losses_df = model_train_test("MLP",                                          
                                         MLPRegression(image_resize, drop_rate=drop_rate), 
                                         "mlp.pth",
                                         train_loader,
                                         validation_loader,
                                         test_loader,
                                         losses_df,
                                         optimizer,
                                         loss_fn,
                                         optim_name,
                                         loss_name,
                                         lr,
                                         num_epochs)
        # CNN
        if load_cnn:
            losses_df = model_train_test("CNN",                                        
                                         CNNetRegression(image_resize, 1, use_batch_norm=batch_size > 1, drop_rate=drop_rate), 
                                         "cnn.pth",
                                         train_loader,
                                         validation_loader,
                                         test_loader,
                                         losses_df,
                                         optimizer,
                                         loss_fn,
                                         optim_name,
                                         loss_name,
                                         lr,
                                         num_epochs)       
        # VGG
        if load_vgg:
            losses_df = model_train_test("VGG16",  
                                         RegressionModel(base_model=vgg.vgg16(in_chans=1, drop_rate=drop_rate)), 
                                         "vgg16.pth",
                                         train_loader,
                                         validation_loader,
                                         test_loader,
                                         losses_df,
                                         optimizer,
                                         loss_fn,
                                         optim_name,
                                         loss_name,
                                         lr,
                                         num_epochs)
        # ResNet
        if load_resnet:
            losses_df = model_train_test("ResNet50",                                          
                                         RegressionModel(base_model=resnet.ResNet(block=resnet.BasicBlock, in_chans=1, layers=(3, 4, 6, 3), drop_rate=drop_rate)), 
                                         "resnet50.pth",
                                         train_loader,
                                         validation_loader,
                                         test_loader,
                                         losses_df,
                                         optimizer,
                                         loss_fn,
                                         optim_name,
                                         loss_name,
                                         lr,
                                         num_epochs)
        # ViT
        if load_vit:
            losses_df = model_train_test("ViT-tiny-patch16", 
                                         RegressionModel(base_model=vision_transformer.VisionTransformer(img_size=image_resize, in_chans=1, patch_size=16, embed_dim=192, depth=12, num_heads=3, drop_rate=drop_rate)), 
                                         "vit-tiny-patch16.pth",
                                         train_loader,
                                         validation_loader,
                                         test_loader,
                                         losses_df,
                                         optimizer,
                                         loss_fn,
                                         optim_name,
                                         loss_name,
                                         lr,
                                         num_epochs)
        # FastViT
        if load_fastvit:
            losses_df = model_train_test("FastViT-SA12",  
                                         RegressionModel(base_model=fastvit.FastVit(in_chans=1, layers=(2, 2, 6, 2), embed_dims=(64, 128, 256, 512), mlp_ratios=(4, 4, 4, 4), token_mixers=("repmixer", "repmixer", "repmixer", "attention"), drop_rate=drop_rate)), 
                                         "fastvit-sa12.pth",
                                         train_loader,
                                         validation_loader,
                                         test_loader,
                                         losses_df,
                                         optimizer,
                                         loss_fn,
                                         optim_name,
                                         loss_name,
                                         lr,
                                         num_epochs)
    else:
        losses_df = pd.read_csv(os.path.join(abs_path, "losses.csv"))

        if load_mlp:
            model_load_test("MLP", 
                            MLPRegression(image_resize, drop_rate=drop_rate),
                            "mlp.pth",
                            test_loader)
        if load_cnn:
            model_load_test("CNN", 
                            CNNetRegression(image_resize, 1, use_batch_norm=batch_size > 1, drop_rate=drop_rate),
                            "cnn.pth",
                            test_loader)
        if load_vgg:
            model_load_test("VGG16", 
                            RegressionModel(base_model=vgg.vgg16(in_chans=1, drop_rate=drop_rate)),
                            "vgg16.pth",
                            test_loader)
        if load_resnet:
            model_load_test("ResNet50", 
                            RegressionModel(base_model=resnet.ResNet(block=resnet.BasicBlock, in_chans=1, layers=(3, 4, 6, 3), drop_rate=drop_rate)),
                            "resnet50.pth",
                            test_loader)
        if load_vit:
            model_load_test("ViT-tiny-patch16", 
                            RegressionModel(base_model=vision_transformer.VisionTransformer(img_size=image_resize, in_chans=1, patch_size=16, embed_dim=192, depth=12, num_heads=3, drop_rate=drop_rate)),
                            "vit-tiny-patch16.pth",
                            test_loader)
        if load_fastvit:
            model_load_test("FastViT-SA12", 
                            RegressionModel(base_model=fastvit.FastVit(in_chans=1, layers=(2, 2, 6, 2), embed_dims=(64, 128, 256, 512), mlp_ratios=(4, 4, 4, 4), token_mixers=("repmixer", "repmixer", "repmixer", "attention"), drop_rate=drop_rate)),
                            "fastvit-sa12.pth",
                            test_loader)

    if plot_losses:
        graph_losses(losses_df)

    if save_losses:
        losses_df.to_csv(os.path.join(abs_path, "losses.csv"))

    # Explainability
    if do_explain:
        explainability(train_set)


if __name__ == "__main__":
    main()