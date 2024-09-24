"""
CS 548 Team Project 2

Authors: Ivan Klevanski Abhiram Yammanuru


Notes:

Place all files (csvs) in the same
directory as the source file

"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os

from scipy import stats

abs_path = os.path.dirname(os.path.abspath(__file__))


def init_dataset():
    """
    Loads the dataset from csv

    **Returns**: Pandas DataFrame for dataset and Metadata
    """

    df = pd.read_csv(os.path.join(abs_path, "Base.csv"))
    mdata = df.dtypes
    return df, mdata


def main():

    df, metadata = init_dataset()

    # TODO: Tasks 1-6

    pass


if __name__ == "__main__":
    main()