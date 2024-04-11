import numpy
import sklearn.feature_extraction.text

import ProbDist
import dataPrep

import argparse
import os
import sys
import numpy as np
import pandas as pd
import itertools
from sklearn.feature_extraction.text import CountVectorizer

import pickle
from sklearn.feature_extraction.text import TfidfVectorizer

DEFAULT_TEST_FILE = '../data/test_data.json'
DEFAULT_TRAIN_FILE = '../data/train_data.json'
DEFAULT_MODEL_FILE = 'naive_bayes.pkl'
CHUNK_SIZE = 1000

EMPTY_FEATURE_VECTOR = dict({'useful': [0, 0], 'funny': [0, 0], 'cool': [0, 0]})

DATA_FORMAT = {

    "stars": [1, 2, 3, 4, 5],
    "useful": int,
    "funny": int,
    "cool": int,
    "text": str

}


# Main running class
class NaiveBayes(object):
    # Control Variables
    testFile = DEFAULT_TEST_FILE
    trainFile = DEFAULT_TRAIN_FILE
    modelFile = DEFAULT_MODEL_FILE
    outFile = DEFAULT_MODEL_FILE
    incFeatures = False
    args: argparse.Namespace

    total_samples = 0

    # Probability distributions for prediction
    p_has_word_category: ProbDist.JointProbDist
    p_stars: ProbDist.ProbDist

    # feature_vectors_dict is a Dictionary of Words used and the total amount of FEATURE they provided to each review
    # followed by the amount of times they contributed. We use this to find the average contribution towards a number
    # of features each word provides, and use this for regression.
    feature_vectors_dict: dict[str: dict[str: [float, int]]]

    def __init__(self):
        self.args = self.parse_args()

        if self.args.train:
            self.train_classifier()
        else:
            self.load_model()

        self.run_test()

    def parse_args(self):
        parser = argparse.ArgumentParser(description='Probabalistic classifier')

        parser.add_argument('--train', action='store_true', required=False,
                            help='Training flag, sets the script to train a model')
        parser.add_argument('--pickle', action='store_true', required=False,
                            help='Pickle flag, sets the script to create a pickle file')
        parser.add_argument('--features', action='store_true', required=False,
                            help='Features flag, sets model to include feature vector calculation in training and '
                                 'prediction', default=False)
        parser.add_argument('--trainFile', type=str, default=DEFAULT_TRAIN_FILE,
                            help='location of training json file, required if training')
        parser.add_argument('--testFile', required=True, type=str, default=DEFAULT_TEST_FILE,
                            help='location testing json file, required')
        parser.add_argument('--modelFile', type=str, default=DEFAULT_MODEL_FILE,
                            help='location of model pickle file, required if not training.')
        parser.add_argument('--outputFile', type=str, default=DEFAULT_MODEL_FILE,
                            help='Location to save output pickle file')

        args = parser.parse_args()

        # Training mode file validation
        if args.train:
            if args.trainFile and not os.path.exists(args.trainFile):
                print('Training data file not found at ' + args.trainFile)
                sys.exit(0)
            else:
                self.trainFile = args.trainFile

            if args.testFile and not os.path.exists(args.testFile):
                print('Testing data file not found at ' + args.testFile + ' A testing data file is required.')
                sys.exit(0)
            else:
                self.testFile = args.testFile

        # PreMade model file validation
        else:
            if args.modelFile and not os.path.exists(args.modelFile):
                print('Model pickle file not found at ' + args.modelFile)
                sys.exit(0)
            else:
                self.modelFile = args.modelFile

        self.incFeatures = args.features
        return args

    def get_feature_vector(self, word: str):
        entry: [float, int] = self.feature_vectors_dict.get(word)
        usefulNorm = 0.0
        funnyNorm = 0.0
        coolNorm = 0.0

        # If we have a feature vector for this word, normalize and store.
        if entry is not None:
            # Normalize vector
            usefulNorm = entry['useful'][0] / float(entry['useful'][1])
            funnyNorm = entry['funny'][0] / float(entry['funny'][1])
            coolNorm = entry['cool'][0] / float(entry['cool'][1])

        return dict({
            'useful': usefulNorm,
            'funny': funnyNorm,
            'cool': coolNorm
        })

    def train_classifier(self):
        print('training classifier...')

        total_null_samples = 0
        total_samples = 0

        self.p_stars = ProbDist.ProbDist('stars',
                                         freqs={1: 1, 2: 1, 3: 1, 4: 1,
                                                5: 1})  # Default freq set to 1 to avoid zero inputs
        self.p_has_word_category = ProbDist.JointProbDist(['hasWord', 'stars'])

        df = pd.read_json(self.trainFile, lines=True, chunksize=CHUNK_SIZE)

        # For each chunk pandas reads from file, handle lines
        for chunk in df:
            print('Training with chunk of ' + str(chunk.shape[0]) + ' samples...')
            null_samples = 0

            # Handle line probabilities
            for i, line in chunk.iterrows():

                # Get feature strengths
                lineStars = line.get('stars')
                lineUseful = float(line.get('useful', 0))
                lineFunny = float(line.get('funny', 0))
                lineCool = float(line.get('cool', 0))

                # Verify line is properly formed
                if lineStars and line['text'] and line['text'].strip() is not '':
                    self.p_stars[lineStars] += 1

                    vectorizer = CountVectorizer(analyzer='words', stop_words='english')
                    vectorizer.fit(line['text'])

                    total_words = float(sum(vectorizer.vocabulary_.values()))

                    # Handle word probabilities within line
                    for word, val in vectorizer.vocabulary_:
                        self.p_has_word_category[word, lineStars] += val

                    # if we're doing feature regression, add feature contributions to words feature vector in dict.
                    if self.incFeatures:
                        for word in vectorizer.vocabulary_.keys():

                            # check we have feature vector for word, create one if needed
                            self.feature_vectors_dict[word] = self.feature_vectors_dict.get(word, EMPTY_FEATURE_VECTOR)

                            # Add word component strengths to vector components, increment occurrences.
                            if lineUseful > 0:
                                self.feature_vectors_dict[word]['useful'][0] += lineUseful * vectorizer.vocabulary_[
                                    word] / total_words
                                self.feature_vectors_dict[word]['useful'][1] += vectorizer.vocabulary_[word]
                            if lineFunny > 0:
                                self.feature_vectors_dict[word]['funny'][0] += lineFunny * vectorizer.vocabulary_[
                                    word] / total_words
                                self.feature_vectors_dict[word]['funny'][1] += vectorizer.vocabulary_[word]
                            if lineCool > 0:
                                self.feature_vectors_dict[word]['cool'][0] += lineCool * vectorizer.vocabulary_[
                                    word] / total_words
                                self.feature_vectors_dict[word]['cool'][1] += vectorizer.vocabulary_[word]

                # Else handle misshapen lines
                else:
                    null_samples += 1

                total_samples += 1

            self.total_samples = total_samples

            print('Done chunk. Found ' + str(null_samples) + ' null samples.')

        print('Normalizing probability matrix...')

        # Iterate over ratings and normalize probabilities.
        for rating in self.p_has_word_category.values('stars'):
            full_count = sum(
                (self.p_has_word_category[word, rating] for word in self.p_has_word_category.values('hasWord')))
            for word in self.p_has_word_category.values('hasWord'):
                self.p_has_word_category[word, rating] = self.p_has_word_category[word, rating] / float(full_count)

        print('Done')

    def load_model(self):
        print('THIS IS NOT IMPLEMENTED YET, ABORTING')
        sys.exit(0)
        # TODO load pickle file here

    def run_test(self):

        test_total = 0
        test_correct = 0
        test_incorrect = 0
        test_null = 0
        residual_absolute_total = 0
        residual_squared_total = 0

        zero_prob = 1.0 / float(self.total_samples)

        print('Running predictions on test file at ' + self.testFile + ' ...')

        # Load chunk from testfile with pandas
        df = pd.read_json(self.testFile, lines=True, chunksize=CHUNK_SIZE)

        # For each chunk of test data, categorize and compare results
        for chunk in df:
            print('Testing with chunk of ' + str(chunk.shape[0]) + ' samples...')

            # For Each line in chunk, run classification and regression
            for line_index, line in chunk.iterrows():

                line_prob = dict({
                    1: self.p_stars[1],
                    2: self.p_stars[2],
                    3: self.p_stars[3],
                    4: self.p_stars[4],
                    5: self.p_stars[5]
                })

                line_rating_ac = int(line['stars'])
                line_text = str(line['text'])
                line_useful_ac = float(line['useful'])
                line_funny_ac = float(line['funny'])
                line_cool_ac = float(line['cool'])
                line_useful = 0.0
                line_funny = 0.0
                line_cool = 0.0

                # Empty string handling and null counter
                if not line_text or line_text == '':
                    test_null += 1

                else:

                    # Count word occurrences in line
                    vectorizer = CountVectorizer(analyzer='words', stop_words='english')
                    vectorizer.fit(line['text'])

                    # For each word in line, multiply probability by word probability for the number of occurrences of word
                    for word in vectorizer.vocabulary_:

                        # Find P (has word | category) for each category.
                        word_prob = dict({
                            1: self.p_has_word_category[word, 1],
                            2: self.p_has_word_category[word, 2],
                            3: self.p_has_word_category[word, 3],
                            4: self.p_has_word_category[word, 4],
                            5: self.p_has_word_category[word, 5]
                        })

                        # Check for zero probabilities, replace them with our small number instead
                        for i in range(5):
                            if word_prob[i + 1] == 0: word_prob[i + 1] = zero_prob

                        # Multiply with word probability for occurrences of word
                        for i in range(vectorizer.vocabulary_[word]):
                            line_prob[1] = line_prob[1] * word_prob[1]
                            line_prob[2] = line_prob[2] * word_prob[2]
                            line_prob[3] = line_prob[3] * word_prob[3]
                            line_prob[4] = line_prob[4] * word_prob[4]
                            line_prob[5] = line_prob[5] * word_prob[5]

                        # if feature regression, add word vector * occurrences
                        if self.incFeatures:
                            feat = self.get_feature_vector(word)
                            line_useful += feat['useful'] * float(vectorizer.vocabulary_[word])
                            line_funny += feat['funny'] * float(vectorizer.vocabulary_[word])
                            line_cool += feat['cool'] * float(vectorizer.vocabulary_[word])

                    # Find highest classification for line
                    max = 0
                    rating = 0
                    for i in range(1,6):
                        if line_prob[i] > max:
                            max = line_prob[i]
                            rating = i

                    if rating is line_rating_ac:
                        test_correct += 1
                    else:
                        test_incorrect += 1

                    # If features enabled, check regression error, add to totals.
                    if self.incFeatures:
                        residual_absolute = (abs(line_useful - line_useful_ac) + abs(line_funny - line_funny_ac)
                                             + abs(line_cool - line_cool_ac))

                        residual_squared = numpy.square(residual_absolute)

                        residual_absolute_total += residual_absolute
                        residual_squared_total += residual_squared

                # Increment test total for completed line
                test_total += 1

def main():
    NaiveBayes()


if __name__ == "__main__":
    main()
