"""
CS 548 Team Project 3

Authors: Ivan Klevanski Abhiram Yammanuru


Notes:

Place all files (csvs) in the same
directory as the source file

"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import copy
import os
import random
import traceback
import timm
import sklearn.metrics as skl_m
import sklearn.model_selection as skl_ms
import torch.nn as nn
import torch.cuda
import albumentations
import albumentations.pytorch
import torch.nn.functional as F

from PIL import Image
from torch.utils.data import Dataset
from timm import utils
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from timm.models import resnet
from lime import lime_image


abs_path = os.path.dirname(os.path.abspath(__file__))

# Params

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
seed = 42
image_resize = 48

num_classes = 1000 # Models from timm use 1000 as default number of classes

# Pytorch-specific methods and classes

class ImageDataSet(Dataset):
    """
    PyTorch Dataset for image data
    """

    preprocessing = albumentations.Compose([
        albumentations.Normalize(max_pixel_value=255, always_apply=True),
        albumentations.pytorch.transforms.ToTensorV2()
    ])

    def __init__(self, images: list[str], labels: list):
        self.images = images
        self.labels = labels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, item):

        image = torch.from_numpy(np.asarray(self.images[item]).reshape(1, image_resize, image_resize))

        # Formating Images, preprocessing, and conversion to tensors
        # image = self.preprocessing(image=image)["image"]

        # Format/adjust image
        label = self.labels[item]
        return image, label


class CNNRegression(nn.Module):
    """
    Generic Convolutional Neural Network for image regression task.<br>
    Note: dim assumes that image is square
    """

    def __init__(self, dim, color_dim, drop_rate=0, use_batch_norm=True):
        super(CNNRegression, self).__init__()

        self.use_batch_norm = use_batch_norm

        fc1_dim = int(((dim - 4) / 2) / 2) # dim - (conv1 kernel_size / conv1 stride) / (maxpool2 stride)

        self.conv1 = nn.Conv2d(in_channels=color_dim, out_channels=96, kernel_size=(4, 4), stride=2)
        self.conv2 = nn.Conv2d(in_channels=96, out_channels=192, kernel_size=(4, 4), stride=1, padding=2)
        self.conv3 = nn.Conv2d(in_channels=192, out_channels=384, kernel_size=(3, 3), stride=1)

        self.mp1 = nn.MaxPool2d(kernel_size=(2, 2), stride=1, padding=1)
        self.mp2 = nn.MaxPool2d(kernel_size=(2, 2), stride=2)
        
        self.bn1 = nn.BatchNorm2d(96)
        self.bn2 = nn.BatchNorm2d(192)
        self.bn3 = nn.BatchNorm2d(384)

        self.fc1 = nn.Linear(in_features=(fc1_dim ** 2) * 384, out_features=8192)
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
        x = self.dropout(x)

        x = self.fc2(x)
        x = self.dropout(x)

        x = self.fc3(x)
        x = self.dropout(x)

        x = self.linear(x)

        return x


class RegressionModel(nn.Module):
    """
    Wrapper class that converts timm image classifier NN models<br>
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


def train_model(
    model: nn.Module, 
    train_loader: DataLoader, 
    validation_loader: DataLoader,
    loss_fn=nn.MSELoss(),
    optim: torch.optim.Optimizer=torch.optim.Adam,
    lr: float=1e-4,
    num_epochs: int=25,
    specifier="",
    loss_str="MSE Loss",
    optim_str="Adam"):
    """
    Trains the model via backpropagation

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

            run_loss += loss.item() * images.size(0)

        return run_loss / (len(data_loader) * data_loader.batch_size)

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

            run_loss += loss.item() * images.size(0)

        return run_loss / (len(data_loader) * data_loader.batch_size)

    if model is not None:
        try:
            print('-' * 20)
            print("Training: " + specifier)
            print('-' * 20 + '\n')

            # Initialization
            optimizer=optim(model.parameters(), lr=lr)

            m_best_weights = copy.deepcopy(model.state_dict())
            m_best_loss = np.inf

            model = model.to(device)

            # Epochs
            for epoch in range(num_epochs):

                print('-' * 20)
                print("Epoch: " + str(epoch + 1) + " out of " + str(num_epochs))
                print('-' * 20)

                epoch_loss = train(train_loader)
                print("\nEpoch Summary: Train Loss: {:.4f}".format(epoch_loss))

                epoch_loss = validate(validation_loader)
                print("\nEpoch Summary: Evaluation Loss: {:.4f}".format(epoch_loss))

                # Store best model weights based on lowest loss
                if epoch_loss <= m_best_loss:
                    m_best_loss = epoch_loss
                    m_best_weights = copy.deepcopy(model.state_dict())

            print("Lowest validation loss: {:.4f}".format(m_best_loss))
            print("Parameters used: Learning Rate: {:.4f}, Number of Epochs: {}, Loss Function: {}, Optimizer: {}\n".format(lr, num_epochs, loss_str, optim_str))
            return m_best_weights, m_best_loss
        except Exception:
            print("[Error]: %s training failed due to an exception, exiting...\n" % specifier)
            print("[Error]: Exception occurred during training")
            traceback.print_exc()
            exit(1)


def test_model(
    model: nn.Module, 
    test_loader: DataLoader, 
    specifier=""):

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


def load_data():
    """
    Loads and initializes all datasets<br>
    **Returns**: dataframe and datasets for regression task
    """

    print("[Info]: Loading data. \n")
    
    df = pd.read_csv(os.path.join(abs_path, "age_gender.csv"))
    df["pixels"] = df["pixels"].apply(lambda img: np.array(img.split(' '), dtype=np.float32))
    df["age"] = df["age"].astype(np.float32)
    df = df.sample(frac=1).reset_index(drop=True)

    train_df, test_df = skl_ms.train_test_split(df, test_size=0.2)

    train_df, validation_df = skl_ms.train_test_split(train_df, test_size=0.15)

    train_set = ImageDataSet(train_df["pixels"].tolist(), train_df["age"].tolist())
    validation_set = ImageDataSet(validation_df["pixels"].tolist(), validation_df["age"].tolist())
    test_set = ImageDataSet(test_df["pixels"].tolist(), test_df["age"].tolist())

    return df, train_set, validation_set, test_set



# Other methods

def EDA():
    # TODO
    pass


def data_preprocessing():
    # TODO: note: may not be necessary since most image augmentations will happen in dataset class
    pass


def explainability():
    # TODO
    pass



def main():

    setup()

    df, train_set, validation_set, test_set = load_data()

    batch_size = 32

    train_loader = DataLoader(train_set, shuffle=True, batch_size=batch_size)
    validation_loader = DataLoader(validation_set, shuffle=True, batch_size=batch_size)
    test_loader = DataLoader(test_set, shuffle=True, batch_size=batch_size)

    train_resnet = True
    train_cnn = False

    # ResNet
    if train_resnet:
        resnet_model = RegressionModel(base_model=resnet.ResNet(block=resnet.BasicBlock, in_chans=1, layers=(3, 4, 6, 3))) # ResNet50

        best_weights, best_loss = train_model(model=resnet_model, train_loader=train_loader, validation_loader=validation_loader, specifier="ResNet50", num_epochs=20)

        torch.save(best_weights, "resnet50.pth")

        resnet_model.load_state_dict(best_weights)

        test_model(model=resnet_model, test_loader=test_loader, specifier="ResNet50")

    # CNN
    if train_cnn:
        cnn_model = CNNRegression(image_resize, 1, use_batch_norm=batch_size > 1)

        best_weights, best_loss = train_model(model=cnn_model, train_loader=train_loader, validation_loader=validation_loader, specifier="CNN", num_epochs=20)

        torch.save(best_weights, "cnn.pth")

        cnn_model.load_state_dict(best_weights)

        test_model(model=cnn_model, test_loader=test_loader, specifier="CNN")

if __name__ == "__main__":
    main()