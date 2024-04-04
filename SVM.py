import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import (
    classification_report,
    recall_score,
    precision_score,
    f1_score,
    roc_curve,
    auc,
    precision_recall_curve,
)
import json

names = ["SVM"]

classifiers = [SVC(probability=True)]

for name, clf in zip(names, classifiers):
    clf_pipe = Pipeline(
        [
            ("tfidf", TfidfVectorizer()),
            (name, clf),
        ]
    )

yelp = pd.read_json('cleaned_data.json', lines=True)
yelp.head()
yelp.info()
yelp['Text Length'] = yelp['text'].apply(len)
yelp.corr(numeric_only=True)

yelp_class_stars = yelp[(yelp['stars'] == 1) | (yelp['stars']==5)]
yelp_class_stars['stars'].value_counts()
yelp_class_stars.info()

X = yelp_class_stars['text']
y = yelp_class_stars['stars']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

clf_pipe.fit(X_train, y_train)

pred = clf_pipe.predict(X_test)
pred_prob = clf_pipe.predict_proba(X_test)[:, 1]

# fpr, tpr, thresholds = roc_curve(y_test, pred_prob)
# precision, recall, thresholds_pr = precision_recall_curve(y_test, pred)

print("\n\n", name, "\n\n")
print(classification_report(y_test, pred))
# print("ROC AUC: ", auc(fpr, tpr))
# print("Precision/Recall AUC: ", auc(precision, recall))
print("\n\n")
