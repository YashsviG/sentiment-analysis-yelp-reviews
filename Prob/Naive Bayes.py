import argparse
import os
import sys

import numpy as np
import pandas as pd
import sklearn
import itertools
from sklearn.naive_bayes import MultinomialNB

import pickle
from sklearn.feature_extraction.text import TfidfVectorizer


# Main running class
class NaiveBayes(object):
    def __init__(self):
        self.args = self.parse_args()
        self.cls = MultinomialNB()
        self.vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5, min_df=5, stop_words="english")

        if self.args.train:
            self.train_classifier()
        else:
            self.load_classifier()

    def parse_args(self):
        parser = argparse.ArgumentParser(description='Probabalistic classifier')

        parser.add_argument('--train', action='store_true', required=False,
                            help='Trainning flag, sets the script to train a model')
        parser.add_argument('--pickle', action='store_true', required=False,
                            help='Pickle flag, sets the script to create a pickle file')
        parser.add_argument('--trainFile', type=str, default='../data/train_data.json',
                            help='training json file, required if training')
        parser.add_argument('--testFile', required=True, type=str, default='../data/test_data.json',
                            help='testing json file, required if training')
        parser.add_argument('--valFile', type=str, default='../data/val_data.json',
                            help='validation json file, required if training')
        parser.add_argument('--modelFile', type=str, default='../models/naive_bayes.pkl',
                            help='model pickle file, required if not training.')
        parser.add_argument('--outputFile', type=str, default='../output/naive_bayes.pkl', help='output pickle file')

        args = parser.parse_args()

        # Training mode file validation
        if args.train:
            if args.trainFile and not os.path.exists(args.trainFile):
                print('Training data file not found at ' + args.trainFile)
                sys.exit(0)
            if args.testFile and not os.path.exists(args.testFile):
                print('Testing data file not found at ' + args.testFile)
                sys.exit(0)
            if args.valFile and not os.path.exists(args.valFile):
                print('Validation data file not found at ' + args.valFile)
                sys.exit(0)

        # PreMade model file validation
        else:
            if args.modelFile and not os.path.exists(args.modelFile):
                print('Model pickle file not found at ' + args.modelFile)
                sys.exit(0)

        return args

    def train_classifier(self):
        print('training classifier...')
        # TODO trainning part goes here

        df = pd.read_json(self.args.output_file, lines=True, chunksize=1000)

    def load_classifier(self):
        print('wow')
        # TODO load pickle file here


def main():
    NaiveBayes()


if __name__ == "__main__":
    main()
