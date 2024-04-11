from collections import Counter

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC


def remove_top_words(X_train, X_test, k):
    # Count the frequency of each word in the training set
    word_counter = Counter(" ".join(X_train).split())
    # Get the top k most frequent words
    top_words = [word for word, _ in word_counter.most_common(k)]
    # Remove the top k most frequent words from the text reviews
    X_train_filtered = [
        " ".join([word for word in review.split() if word not in top_words])
        for review in X_train
    ]
    X_test_filtered = [
        " ".join([word for word in review.split() if word not in top_words])
        for review in X_test
    ]
    return X_train_filtered, X_test_filtered


def experiment_remove_top_words(X_train, X_test, Y_train, Y_test, k):
    # Remove the top k most frequent words
    X_train_filtered, X_test_filtered = remove_top_words(X_train, X_test, k)
    return X_train_filtered, X_test_filtered


def experiment_ngrams(X_train, X_test, Y_train, Y_test):
    count_vectorizer = CountVectorizer(ngram_range=(1, 2))  # Try bi-grams
    X_train_counts = count_vectorizer.fit_transform(X_train)
    X_test_counts = count_vectorizer.transform(X_test)
    return X_train_counts, X_test_counts


def experiment_kernel_optimization(X_train, X_test, Y_train, Y_test):
    param_grid = {"C": [0.1, 1, 10], "kernel": ["linear", "poly", "rbf"]}
    svm_stars = SVC()
    grid_search = GridSearchCV(svm_stars, param_grid, cv=3)
    grid_search.fit(X_train, Y_train)
    svm_stars = grid_search.best_estimator_
    return svm_stars


def experiment_tfidf_vs_count(X_train, X_test, Y_train, Y_test):
    vectorizer = TfidfVectorizer()  # Try TfidfVectorizer
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    return X_train_tfidf, X_test_tfidf


def experiment_class_imbalance_handling(X_train, X_test, Y_train, Y_test):
    param_grid = {
        "C": [0.1, 1, 10],
        "kernel": ["linear", "poly", "rbf"],
        "class_weight": ["balanced", None],
    }
    svm_logistic = SVC()
    grid_search = GridSearchCV(svm_logistic, param_grid, cv=3)
    grid_search.fit(X_train, Y_train)
    svm_logistic = grid_search.best_estimator_
    return svm_logistic


__all__ = [
    experiment_remove_top_words,
    experiment_ngrams,
    experiment_kernel_optimization,
    experiment_tfidf_vs_count,
    experiment_class_imbalance_handling,
]
