import argparse
import json
import pickle

import multiprocess as mp
import pandas as pd
from multiprocess import Pool
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearnex import patch_sklearn

from experiments import (experiment_class_imbalance_handling,
                         experiment_kernel_optimization, experiment_ngrams,
                         experiment_remove_top_words,
                         experiment_tfidf_vs_count)

patch_sklearn()
import numpy as np
from sklearn.metrics import mean_squared_error

training_file = "./data/25k_data.json"
test_file = "./data/test_data2.json"


def save_model(model, filename):
    with open(filename, "wb+") as f:
        pickle.dump(model, f)


def preprocess_text_data(file_path, task="classify"):
    if task == "classify":
        df = pd.read_json(file_path, lines=True)[["stars", "text"]]
    else:
        df = pd.DataFrame(columns=["stars", "text"])

        with open(file_path, "r") as f:
            for line in f:
                data = json.loads(line)
                df = pd.concat([df, pd.DataFrame([data])], ignore_index=True)

    # Check and replace non-string entries with "NOT_A_STRING"
    df["text"] = df["text"].apply(lambda x: str(x) if not isinstance(x, str) else x)

    # Filter out empty strings after conversion
    filtered_df = df.loc[df["text"].apply(lambda x: x.strip() != "")]

    # Ensure 'stars' can be integers between 1 and 5; includes checking for float representations of integers
    filtered_df = filtered_df.loc[
        filtered_df["stars"].apply(
            lambda x: isinstance(x, (int, float)) and x in range(1, 6)
        )
    ]
    filtered_df["text"] = filtered_df["text"].astype(str)
    filtered_df["stars"] = filtered_df["stars"].astype(int)

    # Count the total number of valid rows
    total_valid_rows = len(filtered_df)
    print(total_valid_rows)
    return filtered_df


def parallelize_ngrams(X_train, X_test, y_train, y_test):
    count_vectorizer = CountVectorizer(ngram_range=(1, 1))
    X_train_counts = count_vectorizer.fit_transform(X_train)
    X_test_counts = count_vectorizer.transform(X_test)
    return X_train_counts, X_test_counts


def train_svm(X_train_counts, y_train):
    svm_stars = SVC(kernel="linear")
    svm_stars.fit(X_train_counts, y_train)
    return svm_stars


def stars(experiment1=False, experiment2=False):
    training_data = preprocess_text_data(training_file, task="classify")
    test_data = preprocess_text_data(test_file, task="classify")

    X_train = training_data["text"]
    y_train = training_data["stars"]
    X_test = test_data["text"]
    y_test = test_data["stars"]

    if experiment1:
        # Experiment 1: Using N-grams
        X_train_counts, X_test_counts = experiment_ngrams(
            X_train, X_test, y_train, y_test
        )
    else:
        # Parallelize feature extraction
        mp.set_start_method("fork")
        with mp.Pool(processes=mp.cpu_count()) as pool:
            results = pool.starmap(
                parallelize_ngrams, [(X_train, X_test, y_train, y_test)]
            )
            X_train_counts, X_test_counts = results[0]

    # Model training
    if experiment2:
        svm_stars = experiment_kernel_optimization(
            X_train_counts, X_test_counts, y_train, y_test
        )
    else:
        svm_stars = SVC(kernel="linear")

    svm_stars.fit(X_train_counts, y_train)
    pred = svm_stars.predict(X_test_counts)

    svm_class_report = classification_report(y_test, pred)
    svm_confusion_matrix = confusion_matrix(y_test, pred)
    print("=============== Stars with SVM ==================")
    print(svm_class_report)
    print(svm_confusion_matrix)

    # Save the trained model
    save_model(svm_stars, "./model/svm_stars_model.pkl")


def funny_cool_useful(experiment1=True):
    features = ["cool", "funny", "useful"]
    for target in features:
        training_data = preprocess_text_data(training_file, task="regression")
        test_data = preprocess_text_data(test_file, task="regression")

        X_train = training_data["text"]
        X_test = test_data["text"]

        # Feature extraction
        if experiment1:
            tfidf_vectorizer = TfidfVectorizer()
            X_train = tfidf_vectorizer.fit_transform(X_train)
            X_test = tfidf_vectorizer.transform(X_test)

        # Preparing the labels, assuming labels are in a column named as the value of 'target'
        y_train = training_data[target]
        y_test = test_data[target]

        # Model training
        svm_logistic = SVC(kernel="poly")
        svm_logistic.fit(X_train, y_train)

        y_pred = svm_logistic.predict(X_test)

        rmse = np.sqrt(mean_squared_error(y_test, y_pred))

        print(f"=============== {target} with SVM Regression ==================")
        print(rmse)

        # Save the trained model
        save_model(svm_logistic, f"./model/{target}_svm_model.pkl")


def main():
    parser = argparse.ArgumentParser(description="Run sentiment analysis functions.")
    parser.add_argument("--stars", action="store_true", help="Run stars function")
    parser.add_argument(
        "--funny_cool_useful",
        action="store_true",
        help="Run funny_cool_useful function",
    )
    parser.add_argument(
        "--experiment1", action="store_true", help="Runs with experimen1 1"
    )
    parser.add_argument(
        "--experiment2", action="store_true", help="Runs with experiment 2"
    )

    args = parser.parse_args()

    if args.stars:
        stars(experiment1=args.experiment1, experiment2=args.experiment2)
    if args.funny_cool_useful:
        funny_cool_useful(experiment1=args.experiment1, experiment2=args.experiment2)


if __name__ == "__main__":
    main()
