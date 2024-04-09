import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import classification_report
import json
from sklearn.feature_extraction.text import CountVectorizer
import pickle

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
vectorizer = CountVectorizer(stop_words="english")
X = vectorizer.fit_transform(yelp_class['text'])

y = yelp_class['stars']

clf_stars = SVC(kernel="linear", random_state=42)
clf_stars.fit(X, y)

with open(f"svm.pickle", "wb") as f:
        pickle.dump((clf_stars, vectorizer), f)