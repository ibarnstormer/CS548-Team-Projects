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
import os
import random
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

from timm.models import *
from lime import lime_image

# Params

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
seed = 42
image_resize = 224

num_classes = 1000 # Models from timm use 1000 as default number of classes

# Pytorch-specific methods and classes

class ImageDataSet(Dataset):
    """
    PyTorch Dataset for image data
    """

    preprocessing = albumentations.Compose([
        albumentations.Resize(width=image_resize, height=image_resize),
        albumentations.Normalize(max_pixel_value=255, always_apply=True)
    ])

    def __init__(self, images, labels):
        self.images = images
        self.labels = labels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, item):
        path = self.images[item]

        # Convert to raw byte array
        image = np.asarray(Image.open(path).convert("RGB"))

        # Preprocessing
        image = self.preprocessing(image=image)["image"]

        # Format/adjust image
        out_img = torch.from_numpy(image)
        label = self.labels[item]
        return out_img, label


class CNNRegression(nn.Module):
    """
    Generic Convolutional Neural Network for image regression tasks.<br>
    Note: dim assumes that image is square
    """

    def __init__(self, dim, color_dim, dropout_p=0):
        super(CNNRegression, self).__init__()

        fc1_dim = (((dim - 8) / 4) / 4) - 3 # dim - (conv1 kernel_size / conv1 stride) / (maxpool2 stride) - (conv3 kernel_size)

        self.conv1 = nn.Conv2d(in_channels=color_dim, out_channels=64, kernel_size=(8, 8), stride=4) # 224 -> 216 -> 54
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=192, kernel_size=(8, 8), stride=2, padding="same") # 54 -> 54 (same padding)
        self.conv3 = nn.Conv2d(in_channels=192, out_channels=384, kernel_size=(3, 3), stride=1) # 9 -> 6

        self.maxpool = nn.MaxPool2d(kernel_size=(4, 4), stride=4, padding="same") # 54 -> 54 (same padding)
        self.maxpool2 = nn.MaxPool2d(kernel_size=(4, 4), stride=4) # 54 -> 9

        self.fc1 = nn.Linear(in_features=((fc1_dim ** 2) * 384), out_features=8192)
        self.fc2 = nn.Linear(in_features=8192, out_features=4096)
        self.fc3 = nn.Linear(in_features=4096, out_features=num_classes) # num_classes = 1000
        self.linear = nn.Linear(in_features=num_classes, out_features=1)

        self.dropout = nn.Dropout(p=dropout_p)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.maxpool(x)
        x = self.dropout(x)

        x = self.conv2(x)
        x = F.relu(x)
        x = self.maxpool2(x)
        x = self.dropout(x)

        x = self.conv3(x)
        x = F.relu(x)
        x = self.maxpool(x)
        x = self.dropout(x)

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


def train_model():
    # TODO: Ivan

    def train():
        # TODO: Ivan
        pass

    def validate():
        # TODO: Ivan
        pass



    pass

def test_model():
    pass


# Other methods

def EDA():
    # TODO
    pass

def data_preprocessing():
    # TODO: note: may not be necessary since most image augmentations will happen in dataset class
    pass


def main():

    setup()

    pass

if __name__ == "__main__":
    main()