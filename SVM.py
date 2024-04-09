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

# yelp = pd.read_json('cleaned_data.json', lines=True)
CHUNK_SIZE = 10000
chunks = []
for chunk in pd.read_json('train_data.json',lines=True, chunksize=CHUNK_SIZE):
    chunks.append(chunk)
yelp = pd.concat(chunks, ignore_index=True)

yelp.head()
yelp.info()
yelp['Text Length'] = yelp['text'].apply(len)
yelp.corr(numeric_only=True)

yelp_class = yelp[(yelp['stars'] == 1)| (yelp['stars'] == 2)| (yelp['stars'] == 3)| (yelp['stars'] == 4) | (yelp['stars']==5)]
yelp_class['stars'].value_counts()
yelp_class.info()

X = yelp_class['text']
y = yelp_class['stars']
y2 = yelp_class['useful']

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
