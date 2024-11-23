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
import os
import sys
import surprise
import sklearn.preprocessing as skl_pp
import warnings

from surprise import NMF, SVD, KNNBasic
from tqdm.auto import tqdm

warnings.filterwarnings("ignore")

abs_path = os.path.dirname(os.path.abspath(__file__))

""" ------ Internal Methods ------ """


def load_csvs():
    i = 0

    stat = "[Info]: Loading Datasets: {} / 3"
    sys.stdout.write(f"\r{stat.format(i)}")

    games_df = pd.read_csv(os.path.join(abs_path, "games.csv"))
    i += 1
    sys.stdout.write(f"\r{stat.format(i)}")

    recs_df = pd.read_csv(os.path.join(abs_path, "recommendations.csv"))
    i += 1
    sys.stdout.write(f"\r{stat.format(i)}")

    users_df = pd.read_csv(os.path.join(abs_path, "users.csv"))
    i += 1
    sys.stdout.write(f"\r{stat.format(i)}\n")

    print("[Info]: Finished loading data\n")

    return games_df, users_df, recs_df

def lookup_game_by_encoded_ID(id: int, df: pd.DataFrame):
    return df.loc[df["item"] == id]["title"].iloc[0]


""" ------ Specific Task Methods ------ """

def EDA(games_df: pd.DataFrame, users_df: pd.DataFrame, recs_df: pd.DataFrame):
    print("[Info]: EDA")

    # Housekeeping: Print heads and shapes for each dataframe
    print("[Info]: Head and shape for games dataframe")
    print(games_df.head())
    print(f"Shape: {games_df.shape}")

    print("[Info]: Head and shape for users dataframe")
    print(users_df.head())
    print(f"Shape: {users_df.shape}")

    print("[Info]: Head and shape for recommendations dataframe")
    print(recs_df.head())
    print(f"Shape: {recs_df.shape}")

    # Unique values
    print("[Info]: Unique values per feature for games dataframe:\n{}".format(games_df.nunique()))
    print("[Info]: Unique values per feature for users dataframe:\n{}".format(users_df.nunique()))
    print("[Info]: Unique values per feature for recommendations dataframe:\n{}".format(recs_df.nunique()))

    # Check number of duplicates and missing rows for recommendations
    dup = recs_df.duplicated().sum()
    print(f"[Info]: duplicate reviews: \n{dup}")
    na_num = recs_df.isna().sum()
    print(f"[Info]: missing entry reviews: \n{na_num}")

    # Correlation Heatmap of games and recommendations
    df_games_numeric = games_df.select_dtypes(include=np.number)
    sns.heatmap(df_games_numeric.corr())
    plt.show()

    df_recs_numeric = recs_df.select_dtypes(include=np.number)
    sns.heatmap(df_recs_numeric.corr())
    plt.show()

    # Distributions of variables for games and recommendations
    df_games_numeric.hist(figsize=(25, 25))
    plt.show()

    df_recs_numeric.hist(figsize=(25, 25))
    plt.show()

    # Distribution of positive and negative reviews
    recs_df["is_recommended"].value_counts().plot(kind="bar")
    plt.show()

    plot_boxes = False # These take a long time to run
    if plot_boxes:
        games_recs_df = pd.merge(recs_df, games_df, how="inner", on="app_id")
        
        # Recommended vs price
        print("[Info]: Boxplot for is_recommended and price_final")
        sns.boxplot(x="is_recommended", y="price_final", data=games_recs_df)
        plt.show()

        # Recommended vs user reviews count
        print("[Info]: Boxplot for is_recommended and user_reviews")
        sns.boxplot(x="is_recommended", y="user_reviews", data=games_recs_df)
        plt.show()
        
        # Recommended vs positive ratio
        print("[Info]: Boxplot for is_recommended and positive_ratio")
        sns.boxplot(x="is_recommended", y="positive_ratio", data=games_recs_df)
        plt.show()

    pass

def preprocess_df(df: pd.DataFrame, users_df: pd.DataFrame, games_df: pd.DataFrame):

    print("[Info]: Preprocessing DataFrame")

    # Drop irrelevant columns that won't be used
    print("[Info]: Dropping irrelevant features")
    df.drop(columns=["helpful", "funny", "date", "hours"], inplace=True)

    # Some individuals gave only a small number of reviews, get top 100 reviewers with the most reviews
    print("[Info]: Sampling DataFrame")

    top_most_reviewed_games = df["app_id"].value_counts().nlargest(200)
    top_users = users_df.nlargest(200, "reviews")

    uid_encoder = skl_pp.LabelEncoder()
    top_users["e_u_id"] = uid_encoder.fit_transform(top_users["user_id"])

    df = pd.merge(df, top_users, how="inner", on="user_id")
    df = pd.merge(df, top_most_reviewed_games, how="inner", on="app_id")

    # One-Hot Encode target feature
    print("[Info]: Encoding Features")
    df["is_recommended"] = df["is_recommended"].astype(float)

    # For each game that does not have a review for a particular user set to default value
    print("[Info]: Adding missing reviews")
    no_reviews_list = []
    unique_games = df["app_id"].unique()
    unique_users = df["e_u_id"].unique()

    for aid in tqdm(unique_games):
        for uid in unique_users:
            r = df.loc[(df["e_u_id"] == uid) & (df["app_id"] == aid)]
            if r.size == 0:
                # Other fields aren't important, just need to set the relevant ones: encoded user id, app id, and recommended to 0.5 (between no 0 and yes 1)
                no_reviews_dict = {"e_u_id": uid, 
                                   "app_id": aid, 
                                   "is_recommended": 0.5,
                                   "review_id": 0, 
                                   "user_id": 0,
                                   "products": 0,
                                   "reviews": 0}
                no_reviews_list.append(no_reviews_dict)

    df = pd.concat([df, pd.DataFrame(no_reviews_list)]).reset_index().apply(lambda x: x.sample(frac=1))

    aid_encoder = skl_pp.LabelEncoder()
    df["e_a_id"] = aid_encoder.fit_transform(df["app_id"])

    print("[Info]: Finished Preprocessing\n")

    return df

""" ------ Driver code ------ """

def main():
    # Load data
    games_df, users_df, recs_df = load_csvs()


    # EDA
    do_eda = True
    if do_eda:
        EDA(games_df, users_df, recs_df)

    # Data Preprocessing
    recs_df = preprocess_df(recs_df, users_df, games_df)

    # Create implicit ratings (recommended * average game rating)
    print("[Info]: Feature Engineering")
    recs_df = pd.merge(recs_df, games_df, how="inner", on="app_id")
    recs_df["impl_rating"] = recs_df["is_recommended"] * recs_df["positive_ratio"]
    recs_df["impl_rating"] = recs_df["impl_rating"].astype(np.float32)

    # Model training & evaluation
    """
    Optimizations:

    Cross Validation
    Feature Engineering: convert is_recommended to continuous variable (based on game percent ratio)
    Cross Validation + Feature Engineering

    Note: perform optimizations manually
    """

    as_regression = True # Use Feature Engineering / implicit ratings
    use_cv = True

    print("[Info]: Setting up Scikit-Surprise objects")

    if not as_regression:
        recs_df = recs_df.rename(columns={"e_u_id": "user", "e_a_id": "item", "rating": "rating_old", "is_recommended": "rating"})
    else:
        recs_df = recs_df.rename(columns={"e_u_id": "user", "e_a_id": "item", "rating": "rating_old", "impl_rating": "rating"})

    recs_df.reset_index(inplace=True)

    reader = surprise.Reader(rating_scale=(0, 100) if as_regression else (0, 1))
    dataset = surprise.Dataset.load_from_df(recs_df[["user", "item", "rating"]], reader=reader)

    train_ds, test_ds = surprise.model_selection.train_test_split(dataset, test_size=0.2)

    # Non-Negative Matrix Factorization
    nmf = NMF(n_factors=200)

    if use_cv:
        print("\n[Info]: CV results for NMF:")
        surprise.model_selection.cross_validate(nmf, data=dataset, measures=["mse", "rmse", "mae", "fcp"], cv=10, verbose=True)
    else:
        nmf.fit(train_ds)
        pred = nmf.test(test_ds)

        print("\n[Info]: NMF: MSE of predicted vs original ratings: {:.4f}".format(surprise.accuracy.mse(pred, False)))
        print("[Info]: NMF: RMSE of predicted vs original ratings: {:.4f}".format(surprise.accuracy.rmse(pred, False)))
        print("[Info]: NMF: MAE of predicted vs original ratings: {:.4f}".format(surprise.accuracy.mae(pred, False)))
        print("[Info]: NMF: FCP of predicted vs original ratings: {:.4f}".format(surprise.accuracy.fcp(pred, False)))


    # Singular Value Decomposition
    svd = SVD(n_factors=10)

    if use_cv:
        print("\n[Info]: CV results for SVD:")
        surprise.model_selection.cross_validate(svd, data=dataset, measures=["mse", "rmse", "mae", "fcp"], cv=10, verbose=True)
    else:
        svd.fit(train_ds)
        pred = svd.test(test_ds)

        print("\n[Info]: SVD: MSE of predicted vs original ratings: {:.4f}".format(surprise.accuracy.mse(pred, False)))
        print("[Info]: SVD: RMSE of predicted vs original ratings: {:.4f}".format(surprise.accuracy.rmse(pred, False)))
        print("[Info]: SVD: MAE of predicted vs original ratings: {:.4f}".format(surprise.accuracy.mae(pred, False)))
        print("[Info]: SVD: FCP of predicted vs original ratings: {:.4f}".format(surprise.accuracy.fcp(pred, False)))


    # KNN
    knn = KNNBasic(k=2, verbose=False)

    if use_cv:
        print("\n[Info]: CV results for KNN:")
        surprise.model_selection.cross_validate(knn, data=dataset, measures=["mse", "rmse", "mae", "fcp"], cv=10, verbose=True)
    else:
        knn.fit(train_ds)
        pred = knn.test(test_ds)

        print("\n[Info]: KNN: MSE of predicted vs original ratings: {:.4f}".format(surprise.accuracy.mse(pred, False)))
        print("[Info]: KNN: RMSE of predicted vs original ratings: {:.4f}".format(surprise.accuracy.rmse(pred, False)))
        print("[Info]: KNN: MAE of predicted vs original ratings: {:.4f}".format(surprise.accuracy.mae(pred, False)))
        print("[Info]: KNN: FCP of predicted vs original ratings: {:.4f}".format(surprise.accuracy.fcp(pred, False)))


    # Explainability (Inference runs) -> SHAP and LIME are incompatible with scikit-surprise

    print("\n[Info]: Explainability")
    
    # Get predictions for arbitrary user and item
    nmf_pred = nmf.predict(uid=100, iid=1)
    svd_pred = svd.predict(uid=100, iid=1)
    knn_pred = knn.predict(uid=100, iid=1)

    game_name = lookup_game_by_encoded_ID(1, recs_df)

    print("\n[Info]: NMF: Rating prediction for user 100 for item 1 ({}): {:.4f}".format(game_name, nmf_pred.est))
    print("[Info]: SVD: Rating prediction for user 100 for item 1 ({}): {:.4f}".format(game_name, svd_pred.est))
    print("[Info]: KNN: Rating prediction for user 100 for item 1 ({}): {:.4f}".format(game_name, knn_pred.est))

    # Get top 10 unlisted recommendations for each model for user 100

    all_nmf = np.array([[x.est, x.r_ui, x.iid] for x in nmf.test(test_ds) if x.uid == 100])
    all_svd = np.array([[x.est, x.r_ui, x.iid] for x in svd.test(test_ds) if x.uid == 100])
    all_knn = np.array([[x.est, x.r_ui, x.iid] for x in knn.test(test_ds) if x.uid == 100])

    top10_nmf = np.flip(all_nmf[np.argsort(all_nmf[:, 0])[-10:]], axis=0)
    top10_svd = np.flip(all_svd[np.argsort(all_svd[:, 0])[-10:]], axis=0)
    top10_knn = np.flip(all_knn[np.argsort(all_knn[:, 0])[-10:]], axis=0)

    for k, v in {"NMF": top10_nmf, "SVD": top10_svd, "KNN": top10_knn}.items():
        print(f"\n[Info]: Top 10 predicted ratings for {k} for user 100:")
        for est, _, id in v:
            print("    Rating: {:.4f} for {}".format(est, lookup_game_by_encoded_ID(id, recs_df)))
    
    pass

if __name__ == "__main__":
    main()