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
import sklearn.pipeline as sk_p
import sklearn.preprocessing as sk_pp
import warnings

from scipy import stats

warnings.filterwarnings("ignore")

abs_path = os.path.dirname(os.path.abspath(__file__))

# Plots can take time to render, set to false to speed up execution
plot_fraud_to_feature = False
plot_base_eda = False

def init_dataset():
    """
    Loads the dataset from csv

    **Returns**: Pandas DataFrame for dataset and Metadata
    """

    df = pd.read_csv(os.path.join(abs_path, "Base.csv"))
    mdata = df.dtypes
    return df, mdata

def dataset_description(df: pd.DataFrame, metadata):
    """
    Prints basic information on dataset

    **df**: DataFrame<br>
    **metadata**: Metadata of **df**<br>
    """
    # print metadata
    print("Metadata information:\n")
    for k, v in zip(metadata.index, metadata.array):
        print("{}: {}".format(k, v))

    # Get basic information

    print("===Head===\n")
    print(df.head())

    # Shape
    print("\nShape: {}".format(df.shape))

    # NaN rows
    print("\nNumber of incomplete / NaN records:\n{}".format(df.isnull().sum()))

    # Duplicated rows
    print("\nNumber of duplicated records: {}".format(df.duplicated().sum()))


def data_preprocessing(df: pd.DataFrame):
    """
    Preprocesses the dataset's features 
    """

    # One Hot Encoding:
    df_non_numeric_col = df.select_dtypes(include=object).columns
    df_numeric = df.select_dtypes(include=np.number)

    ohe = sk_pp.OneHotEncoder(handle_unknown="ignore")
    ohe_features = ohe.fit_transform(df[df_non_numeric_col])

    ohe_feat_df = pd.DataFrame(ohe_features.toarray(), columns=ohe.get_feature_names_out(), dtype=int)
    df_numeric[ohe_feat_df.columns] = ohe_feat_df

    df = df_numeric

    # Other preprocessing

    return df



def EDA(df: pd.DataFrame):

    # Get dataset-specific information

    # Distribution between Fraudulent records and Non-fraudulent records:
    if plot_base_eda:
        vcs = df["fraud_bool"].value_counts()
        vcs.plot(kind="bar")
        for idx in vcs.index:
            plt.text(idx, vcs[idx], vcs[idx], ha="center", va="center")
        
        plt.show()

    # Correlation Heatmap + raw printed
    df_numeric = df.select_dtypes(include=np.number)

    if plot_base_eda:
        sns.heatmap(df_numeric.corr())
        plt.show()

    # Prrinted corr of fraud_bool w.r.t. feature
    print("Correlations to target variable:\n")
    for col in df_numeric.columns:
        print("{}: {}".format(col, df[col].corr(df["fraud_bool"])))

    # Distributions
    if plot_base_eda:
        df_numeric.hist(figsize=(25, 25))
        plt.show()

    """ 
    (Double check features to see if siginficant, else remove):
    - foreign_request

    """

    print("Unique values per feature:\n{}".format(df.nunique()))

    # Checking significant / interesting columns

    # Significant (more fraudsters from foreign requests)
    if plot_fraud_to_feature:
        sns.barplot(x=df["foreign_request"], y=df["fraud_bool"])
        plt.show()

    # More fraudsters in month 7
    if plot_fraud_to_feature:
        sns.barplot(x=df["month"], y=df["fraud_bool"])
        plt.show()
    
    # Could be useful (less likely to commit fraud on lower risk score accounts)
    if plot_fraud_to_feature:
        sns.scatterplot(x=df["credit_risk_score"], y=df["fraud_bool"])
        plt.show()
    
    # Could be useful (CC most likely)
    if plot_fraud_to_feature:
        sns.barplot(x=df["employment_status"], y=df["fraud_bool"])
        plt.show()

    # Most fraudulent accounts use Windows
    if plot_fraud_to_feature:
        sns.barplot(x=df["device_os"], y=df["fraud_bool"])
        plt.show()

    # AC payment type has the majority of fraudulent accounts
    if plot_fraud_to_feature:
        sns.barplot(x=df["payment_type"], y=df["fraud_bool"])
        plt.show()
    
    # Bigger gap between 50-60 days for fraudulent accounts
    if plot_fraud_to_feature:
        sns.scatterplot(x=df["days_since_request"], y=df["fraud_bool"])
        plt.show()

    # Useful (fraudsters target accounts whose original owners are very old)
    if plot_fraud_to_feature:
        sns.barplot(x=df["customer_age"], y=df["fraud_bool"])
        plt.show()

    """
    Omit the following features:
    device_fraud_count: (all records are the same value)
    """

    df.drop(columns=["device_fraud_count"], inplace=True)


def main():

    df, metadata = init_dataset()

    # TODO: Tasks 1-6

    dataset_description(df, metadata)

    EDA(df)

    df = data_preprocessing(df)

    pass


if __name__ == "__main__":
    main()