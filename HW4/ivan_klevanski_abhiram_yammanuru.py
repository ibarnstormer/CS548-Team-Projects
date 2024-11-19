"""
CS 548 Team Project 4

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
import torchsummary
import torchvision
import albumentations
import albumentations.pytorch
import torch.nn.functional as F
import warnings

from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from captum.attr import LayerGradCam


""" ------ Datasets and Models ------ """

class SteamGamesDataset(Dataset):
    def __init__(self, df: pd.DataFrame):
        self.data = df

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data["e_uid"][idx], self.data["e_gid"][idx], self.data["f_rating"][idx]


class MLPNet(nn.Module):
    """
    Multi-Layer Perceptron for Neural Collaborative Filtering\n
    Based on: https://arxiv.org/abs/1708.05031
    """

    def __init__(self, num_items, num_users, embedding_dim, drop_rate: float):
        super(MLPNet, self).__init__()

        self.ilv_embedding = nn.Embedding(num_items, embedding_dim)
        self.ulv_embedding = nn.Embedding(num_users, embedding_dim)

        mlp_input_size = embedding_dim * 2

        self.mlp = nn.Sequential(
            nn.Linear(mlp_input_size, int(mlp_input_size / 2)),
            nn.ReLU(),
            nn.Dropout(drop_rate),
            nn.Linear(int(mlp_input_size / 2), int(mlp_input_size / 4)),
            nn.ReLU(),
            nn.Dropout(drop_rate),
            nn.Linear(int(mlp_input_size / 4), int(mlp_input_size / 8)),
            nn.ReLU(),
            nn.Dropout(drop_rate),
            nn.Linear(int(mlp_input_size / 8), int(mlp_input_size / 16)),
            nn.ReLU(),
            nn.Dropout(drop_rate),
            nn.Linear(int(mlp_input_size / 16), int(mlp_input_size / 32)),
            nn.ReLU(),
            nn.Dropout(drop_rate)
        )

        self.linear = nn.Linear(int(mlp_input_size / 32), int(mlp_input_size / 48))
        self.sigmoid = nn.Sigmoid()

    def forward(self, item_vec, user_vec):
        concat_embed = torch.concat((self.ilv_embedding(item_vec), self.ulv_embedding(user_vec)), -1)
        output = self.mlp(concat_embed)

        return self.sigmoid(self.linear(output))


class GMFNet(nn.Module):
    """
    Generalized Matrix Factorization\n
    Based on: https://arxiv.org/abs/1708.05031
    """

    def __init__(self, num_items, num_users, embedding_dim):
        super(GMFNet, self).__init__()

        self.ilv_embedding = nn.Embedding(num_items, embedding_dim)
        self.ulv_embedding = nn.Embedding(num_users, embedding_dim)

        self.linear = nn.Linear(embedding_dim, embedding_dim)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, item_vec, user_vec):
        dp = self.ilv_embedding(item_vec) * self.ulv_embedding(user_vec)

        return self.sigmoid(self.linear(dp))


class NeuMFNet(nn.Module):
    """
    Neural Matrix Factorization Model\n
    Based on: https://arxiv.org/abs/1708.05031
    """

    def __init__(self, pretrained_mlp: MLPNet, pretrained_gmf: GMFNet, mlp_dim: int, gmf_dim: int):
        super(NeuMFNet, self).__init__()

        nmf_linear_input = mlp_dim + gmf_dim

        self.mlp = pretrained_mlp
        self.gmf = pretrained_gmf

        self.linear = nn.Linear(nmf_linear_input, 1)
        self.sigmoid = nn.Sigmoid()

    
    def forward(self, item_vec, user_vec):
        mlp = self.mlp(item_vec, user_vec)
        gmf = self.gmf(item_vec, user_vec)
        concat_embed = torch.concat((gmf, mlp), -1)

        return self.sigmoid(self.linear(concat_embed))
    

""" ------ Internal Methods ------ """

# Deep Learning models

def train_mlp():
    pass

def test_mlp():
    pass

def train_gmf():
    pass

def test_gmf():
    pass

def train_neumf():
    pass

def test_neumf():
    pass

def neumf_inference():
    pass

# Regular ML models

def train_xgboost():
    pass

def test_xgboost():
    pass

def xgboost_inference():
    pass

def train_KNN():
    pass

def test_KNN():
    pass

def KNN_inference():
    pass


""" ------ Specific Task Methods ------ """

def EDA():
    pass

def preprocess_df():
    pass

def model_training():
    pass

def evaluation():
    pass

""" ------ Driver code ------ """

def main():
    pass




if __name__ == "__main__":
    main()