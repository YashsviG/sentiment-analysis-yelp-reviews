import json
from Review import Review
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import SVC
import pickle
import argparse
from sklearn.metrics import classification_report
import os
import os.path as osp
import pickle
import pandas as pd

techniques = ["bert", "dtc", "svm"]

def main():
    parser = argparse.ArgumentParser(
        prog="Project 2 - Model Training Program",
        description="Program to train models for predicting Yelp review stars",
    )
    parser.add_argument(
        "technique", choices=techniques, help="Which technique to train the model for."
    )
    parser.add_argument(
        "-t",
        dest="enable_tuning",
        default=False,
        action="store_true",
        help="Optional flag to enable hyperparameter tuning during training",
    )
    parser.add_argument(
        "--batch-size",
        dest="batch_size",
        type=int,
        default=5000,
        help="Number of reviews to process in each batch.",
    )
    parser.add_argument(
        "--max-train-entries",
        dest="max_train_entries",
        type=int,
        default=10000,
        help="Maximum number of reviews to train the model with.",
    )
    args = parser.parse_args()

    batch_num = 1
    total_processed = 0
    reviews = []  # Initialize reviews list outside the loop
    
    with open("train_data.json", "r", encoding="utf-8") as train:
        print("Pre-processing data from training dataset...\n")
        
        for _ in range(args.max_train_entries):
            try:
                entry = next(train)
                review = Review(json.loads(entry))
                reviews.append(review)
                total_processed += 1  # Increment total_processed
                
                # if total_processed % args.batch_size == 0 or total_processed == args.max_train_entries:
                    # Train the model for the current batch
                train_batch(reviews, args.technique, batch_num)
                batch_num += 1
                reviews = []  # Reset reviews list for the next batch
                
                # if total_processed == args.max_train_entries:
                #     break
                
            except StopIteration:
                break
            except Exception as e:
                print("Error:", e)

        combine_pickle_files('./', './final.pickle')

    return 0

def train_batch(reviews, technique, batch_num):
    # Split the data into text and labels
    text = [review.text for review in reviews]
    stars = [review.stars for review in reviews]

    vectorizer = CountVectorizer(stop_words="english")
    X = vectorizer.fit_transform(text)

    # Train SVM model to predict star ratings
    clf_stars = SVC(kernel="linear", random_state=42)
    clf_stars.fit(X, stars)

    # Save the trained model and vectorizer to an output file
    with open(f"{technique}_batch_{batch_num}.pickle", "wb") as f:
        pickle.dump((clf_stars, vectorizer), f)

    # Convert the test data into a bag-of-words representation using the vectorizer
    X_test = vectorizer.transform(text)

    # Predict the star ratings using the trained model
    y_pred_stars = clf_stars.predict(X_test)

    # Print the classification report to evaluate the model's performance
    print(f"Batch {batch_num} - Star ratings:")
    print(classification_report(stars, y_pred_stars))

def combine_pickle_files(directory_path, output_file):
    combined_df = pd.DataFrame()

    for file_name in os.listdir(directory_path):
        if file_name.endswith(".pickle"):
            file_path = osp.join(directory_path, file_name)
            with open(file_path, "rb") as f:
                content = pickle.load(f)
                if isinstance(content, pd.DataFrame):
                    combined_df = pd.concat([combined_df, content], ignore_index=True)

    with open(output_file, "wb") as out:
        pickle.dump(combined_df, out, protocol=pickle.HIGHEST_PROTOCOL)

if __name__ == "__main__":
    main()
