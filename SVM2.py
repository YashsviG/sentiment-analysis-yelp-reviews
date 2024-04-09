import json
from Review import Review
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import SVC, LinearSVC
import pickle
import argparse
from sklearn.metrics import classification_report
from sklearn.metrics import mean_squared_error, r2_score


techniques = ['bert', 'dtc', 'svm']

def main():
    parser = argparse.ArgumentParser(
                    prog='Project 2 - Model Training Program',
                    description='Program to train models for predicting Yelp review stars')
    parser.add_argument('technique', choices=techniques,
                    help='Which technique to train the model for.')
    parser.add_argument('-t', dest='enable_tuning', default=False, action='store_true',
                    help='Optional flag to enable hyperparameter tuning during training')
    parser.add_argument('--max-train', dest='max_train_entries', type=int, default=10000,
                    help='Maximum number of reviews to train the model with.')
    args = parser.parse_args()

    reviews = []
    # Will need to grab file name from the user
    with open('train_data.json', 'r', encoding="utf-8") as train:
        print('Pre-processing data from training dataset...\n')
        total = 0
        processed = 0
        skipped = 0
        for entry in train:
            if total >= args.max_train_entries: break
            total += 1
            try:
                review = Review(json.loads(entry))
                reviews.append(review)
                processed += 1
            except:
                skipped += 1

    if args.technique == 'bert':
        pass
    elif args.technique == 'dtc':
        pass
    elif args.technique == 'svm':
        # SVM code
        # Split the data into text and labels
        text = [review.text for review in reviews]
        stars = [review.stars for review in reviews]
        useful = [review.useful for review in reviews]
        funny = [review.funny for review in reviews]
        cool = [review.cool for review in reviews]

        # Convert the text into a bag-of-words representation
        stopWords = ['english', 'french', 'spanish', 'russian']
        vectorizer = CountVectorizer(stop_words= stopWords)
        X = vectorizer.fit_transform(text)

        # Train SVM models to predict star ratings, usefulness, funniness, and coolness
        clf_stars = SVC(kernel='linear', random_state=42)
        clf_stars.fit(X, stars)

        clf_useful_l1= LinearSVC(penalty='l1', dual=False, random_state=42)
        clf_useful_l1.fit(X, useful)

        clf_funny_l1= LinearSVC(penalty='l1', dual=False, random_state=42)
        clf_funny_l1.fit(X, funny)

        clf_cool_l1= LinearSVC(penalty='l1', dual=False, random_state=42)
        clf_cool_l1.fit(X, cool)

        # Save the trained models and vectorizer to output files
        with open('svm.pickle', 'wb') as f:
            pickle.dump((clf_stars, clf_useful_l1, clf_funny_l1, clf_cool_l1, vectorizer), f)
        with open('svm.pickle', 'rb') as f:
            clf_stars, clf_useful_l1, clf_funny_l1, clf_cool_l1, vectorizer = pickle.load(f)

        # Convert the test data into a bag-of-words representation using the vectorizer
        X_test = vectorizer.transform(text)

        # Predict the ratings and review attributes using the trained models
        y_pred_stars = clf_stars.predict(X_test)
        y_pred_useful = clf_useful_l1.predict(X_test)
        y_pred_funny = clf_funny_l1.predict(X_test)
        y_pred_cool = clf_cool_l1.predict(X_test)
        
        # Print the classification reports to evaluate the models' performance
        print('Star ratings:')
        print(classification_report(stars, y_pred_stars))

        print('Usefulness:')
        print('Mean Squared Error:', mean_squared_error(useful, y_pred_useful))
        print('R-squared:', r2_score(useful, y_pred_useful))

        print('Funniness:')
        print('Mean Squared Error:', mean_squared_error(funny, y_pred_funny))
        print('R-squared:', r2_score(funny, y_pred_funny))

        print('Coolness:')
        print('Mean Squared Error:', mean_squared_error(cool, y_pred_cool))
        print('R-squared:', r2_score(cool, y_pred_cool))
        pass
    return 0


    return 0

if __name__ == '__main__':
    main()