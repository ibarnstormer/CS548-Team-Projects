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
from sklearn.metrics import accuracy_score
from sklearn.discriminant_analysis import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.metrics import f1_score
from sklearn.metrics import matthews_corrcoef
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

def training(df: pd.DataFrame):
    """
    Splits the dataset into features (X) and labels (y),
    and creates training, validation, and testing datasets.
    """
    # Using all features except fraud_bool and device_fraud_count
    X = df.drop(['fraud_bool'], axis=1) 

    # Extracting fraud_bool as the label
    y = df['fraud_bool']  # Target variable

    # Splitting into training, testing, and validation datasets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.125, random_state=42) # 0.25 * 0.8 = 0.2

    # Return variables needed for further processing
    return X, y, X_train, X_test, y_train, y_test

def logReg(X_train, y_train, X_test, y_test):
    """
    Performs logistic regression and returns accuracy and confusion matrix.
    """
    # Scaling the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)  

    # Logistic Regression
    logregmodel = LogisticRegression()
    logregmodel.fit(X_train_scaled, y_train)

    # Predict on the testing data
    y_pred = logregmodel.predict(X_test_scaled)

    # Print accuracy
    print("Logistic Regression Accuracy:", accuracy_score(y_test, y_pred))

    # K-fold cross-validation
    num_folds = 5
    kf = KFold(n_splits=num_folds, shuffle=True, random_state=42)
    cross_val_scores = cross_val_score(logregmodel, X_train_scaled, y_train, cv=kf)

    print("Logistic Regression Cross-validation scores:", cross_val_scores)
    mean_accuracy = cross_val_scores.mean()
    print("Logistic Regression Mean accuracy:", mean_accuracy)

    #F1 scores 
    f1 = f1_score(y_test, y_pred)
    print("Logistic Regression mean F1 Score ", y_pred)

    #MCC scores 
    mcc = matthews_corrcoef(y_test, y_pred)
    print("Logistic Regression MCC ", mcc)

    # Confusion matrix
    confusionMatrix = confusion_matrix(y_test, y_pred)

    # Plot the confusion matrix
    plt.figure(figsize=(7, 5))
    sns.heatmap(confusionMatrix, annot=True, cmap="viridis", fmt="d", cbar=False)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix for Logistic Regression")
    plt.show()

    #Precision and Recall 
    tn, fp, fn, tp = confusion_matrix.ravel()
    p = tp / (tp + fp)
    r = tp / (tp + fn)

    print("Logistic Regression Precision score = ", p)
    print("Logistic regression Recall score = ", r)

    
def adaBoost(X_train, y_train, X_test, y_test):
    
    """
    Performs adaboost and returns accuracy, f1 score, precision, recall, mccscore and confusion matrix.
    """
    # Scaling the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)  

    # Logistic Regression
    adamodel = AdaBoostClassifier()
    adamodel.fit(X_train_scaled, y_train)

    # Predict on the testing data
    y_pred = adamodel.predict(X_test_scaled)

    # Print accuracy
    print("ADA Boost Accuracy:", accuracy_score(y_test, y_pred))

    # K-fold cross-validation
    num_folds = 5
    kf = KFold(n_splits=num_folds, shuffle=True, random_state=42)
    cross_val_scores = cross_val_score(adamodel, X_train_scaled, y_train, cv=kf)

    print("ADA Boost Cross-validation scores:", cross_val_scores)
    mean_accuracy = cross_val_scores.mean()
    print("ADA Boost Mean accuracy:", mean_accuracy)

    #F1 scores 
    f1 = f1_score(y_test, y_pred)
    print("ADA Boost F1 Score ", y_pred)

    #MCC scores 
    mcc = matthews_corrcoef(y_test, y_pred)
    print("ADA Boost MCC ", mcc)

    # Confusion matrix
    confusionMatrix = confusion_matrix(y_test, y_pred)

    # Plot the confusion matrix
    plt.figure(figsize=(7, 5))
    sns.heatmap(confusionMatrix, annot=True, cmap="viridis", fmt="d", cbar=False)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix for ADA Boost")
    plt.show()

    #Precision and recall 
    tn, fp, fn, tp = confusion_matrix.ravel()
    p = tp / (tp + fp)
    r = tp / (tp + fn)

    print("ADA Boost Precision score = ", p)
    print("ADA Boost Recall score = ", r)
    
def knn(X_train, X_test, y_train, y_test, n_neighbors=5):
    """
    Performs K-Nearest Neighbors classification and returns accuracy, F1, MCC, precision, recall, and confusion matrix.
    """
    # Scaling the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)  

    # K-Nearest Neighbors
    knn_model = KNeighborsClassifier(n_neighbors=n_neighbors)
    knn_model.fit(X_train_scaled, y_train)

    # Predict on the testing data
    y_pred = knn_model.predict(X_test_scaled)

    # Print accuracy
    print("KNN Accuracy:", accuracy_score(y_test, y_pred))

    # K-fold cross-validation
    num_folds = 5
    kf = KFold(n_splits=num_folds, shuffle=True, random_state=42)
    cross_val_scores = cross_val_score(knn_model, X_train_scaled, y_train, cv=kf)

    print("KNN Cross-validation scores:", cross_val_scores)
    mean_accuracy = cross_val_scores.mean()
    print("KNN Mean accuracy:", mean_accuracy)

    # F1 score
    f1 = f1_score(y_test, y_pred)
    print("KNN F1 Score:", f1)

    # MCC (Matthews correlation coefficient)
    mcc = matthews_corrcoef(y_test, y_pred)
    print("KNN MCC:", mcc)

    # Confusion matrix
    confusionMatrix = confusion_matrix(y_test, y_pred)

    # Plot the confusion matrix
    plt.figure(figsize=(7, 5))
    sns.heatmap(confusionMatrix, annot=True, cmap="viridis", fmt="d", cbar=False)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix for KNN")
    plt.show()

    # Precision and Recall
    tn, fp, fn, tp = confusionMatrix.ravel()
    precision = tp / (tp + fp)  # Precision
    recall = tp / (tp + fn)     # Recall

    print("KNN Precision score =", precision)
    print("KNN Recall score =", recall)


def main():
    df, metadata = init_dataset()

    dataset_description(df, metadata)

    EDA(df)

    # Preprocess data
    df = data_preprocessing(df)

    # Train and test split
    X, y, X_train, X_test, y_train, y_test = training(df)

    #ADA boost 
    adaBoost(X_train, y_train, X_test, y_test)

    # Logistic Regression
    logReg(X_train, y_train, X_test, y_test)

    #KNN
    knn(X_train, y_train, X_test, y_test, 5)

    pass


if __name__ == "__main__":
    main()
