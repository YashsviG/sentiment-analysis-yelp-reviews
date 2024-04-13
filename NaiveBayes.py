import ProbDist

import argparse
import os
import sys
import time
import datetime
import dill as pickle
# import pickle
import numpy
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

DEFAULT_TEST_FILE = 'data/val_data.json'
DEFAULT_TRAIN_FILE = 'data/train_data.json'
DEFAULT_MODEL_FILE = 'pickled-models/naive_bayes.pkl'
TRAINING_CHUNK_SIZE = 10000  # will always train on at least one full chunk
MAX_TRAINING = 100000
TESTING_CHUNK_SIZE = 100000  # will always test on at least one full chunk
MAX_TESTING = 300000

USE_CUSTOM_NORMALIZER = True

# Empty versions of feature vector for creating new entries in the dictionary.
# Probably not easier than just making a new class for it.
EMPTY_FEATURE_VECTOR = dict({'useful': [0, 0], 'funny': [0, 0], 'cool': [0, 0]})

# Data format for dataframes given by pandas. Used for handling dataframes.
DATA_FORMAT = {
    "stars": [1, 2, 3, 4, 5],
    "useful": int,
    "funny": int,
    "cool": int,
    "text": str
}

def parse_args( parser: argparse.ArgumentParser):
    """Custom Argument parser for handling construction of the class without inputs. Includes basic validation."""
    parser.add_argument('--train', action='store_true', default=False,
                        help='Training flag, sets the script to train a model')
    parser.add_argument('--pickle', action='store_true', required=False,
                        help='Pickle flag, sets the script to create a pickle file')
    parser.add_argument('--noFeatures', action='store_false', required=False,
                        help='Features flag, sets model to include feature vector calculation in training and '
                             'prediction', default=True)
    parser.add_argument('--training_file', type=str, default=DEFAULT_TRAIN_FILE,
                        help='location of training json file, required if training')
    parser.add_argument('--test_file', required=True, type=str, default=DEFAULT_TEST_FILE,
                        help='location testing json file, required')
    parser.add_argument('--trained_model', type=str, default=DEFAULT_MODEL_FILE,
                        help='location of model pickle file, required if not training.')
    parser.add_argument('--out_file', type=str, default=DEFAULT_MODEL_FILE,
                        help='Location to save output pickle file')

    args = parser.parse_args()

    # Testing file validation
    if args.test_file and not os.path.isfile(args.test_file):
        print('Testing data file not found at ' + args.test_file + ' A testing data file is required.')
        sys.exit(0)

    # Training mode file validation
    if args.train:
        if not args.training_file:
            print('Training File required for training model.')
            sys.exit(0)

        if args.training_file and not os.path.isfile(args.training_file):
            print('Training data file not found at ' + args.training_File)
            sys.exit(0)

    # PreMade model file validation
    else:
        if args.trained_model and not os.path.isfile(args.trained_model):
            print('Model pickle file not found at ' + args.trained_model)
            sys.exit(0)

    return args


# Main running class
class NaiveBayes(object):
    feature_vectors_dict = None
    p_has_word_category = None
    p_stars = None
    testFile = DEFAULT_TEST_FILE
    trainFile = DEFAULT_TRAIN_FILE
    modelFile = DEFAULT_MODEL_FILE
    outFile = DEFAULT_MODEL_FILE
    custom_args = False
    run_training = False
    run_pickle = True
    incFeatures = True
    args: argparse.Namespace
    total_samples = 0

    def __init__(self, input_args: argparse.Namespace, custom_args=False):
        """Constructor for Naive Bayes classifier. Takes input arguments from an argeParser, assumes arguments are
        valid. If calling with your own argParser, make sure custom_args is set to True."""
        # Control Variables
        self.feature_vectors_dict = None
        self.p_has_word_category = None
        self.p_stars = None
        self.testFile = DEFAULT_TEST_FILE
        self.trainFile = DEFAULT_TRAIN_FILE
        self.modelFile = DEFAULT_MODEL_FILE
        self.outFile = DEFAULT_MODEL_FILE
        self.custom_args = custom_args
        self.run_training = False
        self.run_pickle = True
        self.incFeatures = True
        self.args: argparse.Namespace
        self.total_samples = 0

        # Probability distributions for prediction
        self.p_has_word_category: ProbDist.JointProbDist = None
        self.p_stars: ProbDist.ProbDist
        self.feature_vectors_dict: dict[str: dict[str: [float, int]]]
        # feature_vectors_dict is a Dictionary of Words used and the total amount of FEATURE they provided to each
        # review followed by the amount of times they contributed. We use this to find the average contribution
        # towards a number of features each word provides, and use this for regression. Why did I do this instead of
        # using another ProbDist? who knows.

        self.args = input_args
        self.handle_args()

        if self.run_training:
            self.train_classifier()
            if self.run_pickle:
                self.save_model()

        else:
            self.load_model()

        self.run_test()

    def handle_args(self):
        """Function to handle custom arguments being passed to the class from an argeParser. Assumes arguments are
        valid and correctly labeled."""

        # If handling passed args
        if self.custom_args:
            self.incFeatures = True

            if self.args.training_file:
                self.run_training = True

        # if handling in file args
        else:
            self.run_training = self.args.train

            if self.args.noFeatures:
                self.incFeatures = True
            else:
                self.incFeatures = False

        self.run_pickle = self.args.pickle
        self.trainFile = self.args.training_file
        self.testFile = self.args.test_file
        self.outFile = self.args.out_file

    def get_feature_vector(self, word: str):
        """Function to get a Feature vector (Dictionary) for a particular word. Returns a zero vector if the word is
        not in dictionary."""
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
        """Function to run training procedures for the Naive Bayes classifier. Overwrites any existing probability
        data. Takes no inputs, relies on stored class variables. Loads data from location given in self.trainFile."""
        print('Training classifier...')

        # INITIALIZATION
        train_start = datetime.datetime.now()
        total_null_samples = 0
        total_samples = 0

        word_totals = dict({1: 0, 2: 0, 3: 0, 4: 0, 5: 0})

        self.p_stars = ProbDist.ProbDist('stars',
                                         freqs={1: 1, 2: 1, 3: 1, 4: 1, 5: 1})  # Default set to 1 to avoid zero inputs
        self.p_has_word_category = ProbDist.JointProbDist(['hasWord', 'stars'])

        # Workaround for bugs with inserting into empty dictionary, used most common words.
        self.feature_vectors_dict: dict[str: dict[str: [float, int]]] = dict(
            {'the': EMPTY_FEATURE_VECTOR, 'be': EMPTY_FEATURE_VECTOR, }
        )

        # BUILDING WORD FREQUENCIES
        df = pd.read_json(self.trainFile, lines=True, chunksize=TRAINING_CHUNK_SIZE)

        # For each chunk pandas reads from file, handle lines
        for chunk in df:

            # Training early stop for faster testing, edit number at top of file.
            if (total_samples >= MAX_TRAINING) and (MAX_TRAINING != 0):
                print("Exceeded maximum number of samples.")
                break

            print('Training with chunk of ' + str(chunk.shape[0]) + ' samples...')

            null_samples = 0
            start = time.time()

            # Handle line probabilities
            for i, line in chunk.iterrows():

                # Get feature strengths
                lineStars = line.get('stars')
                lineUseful = float(line.get('useful', 0))
                lineFunny = float(line.get('funny', 0))
                lineCool = float(line.get('cool', 0))

                # Verify line is properly formed
                if lineStars and line['text'] and line['text'].strip() != '':
                    self.p_stars[lineStars] += 1

                    vectorizer = CountVectorizer(analyzer='word', stop_words='english')
                    try:
                        vectorizer.fit([line['text']])

                        total_words = float(sum(vectorizer.vocabulary_.values()))

                        # Handle word probabilities within line
                        for word in vectorizer.vocabulary_:
                            self.p_has_word_category[word, lineStars] += vectorizer.vocabulary_[word]
                            word_totals[lineStars] += vectorizer.vocabulary_[word]

                        # if we're doing feature regression, add feature contributions to words feature vector in dict.
                        if self.incFeatures:
                            for word in vectorizer.vocabulary_.keys():

                                # check we have feature vector for word, create one if needed
                                self.feature_vectors_dict[word] = self.feature_vectors_dict.get(word,
                                                                                                EMPTY_FEATURE_VECTOR)

                                # Add word component strengths to vector components, increment occurrences.
                                if lineUseful > 0:
                                    self.feature_vectors_dict[word]['useful'][0] += lineUseful * vectorizer.vocabulary_[
                                        word] / max(total_words, 1.0)
                                    self.feature_vectors_dict[word]['useful'][1] += vectorizer.vocabulary_[word]
                                if lineFunny > 0:
                                    self.feature_vectors_dict[word]['funny'][0] += lineFunny * vectorizer.vocabulary_[
                                        word] / max(total_words, 1.0)
                                    self.feature_vectors_dict[word]['funny'][1] += vectorizer.vocabulary_[word]
                                if lineCool > 0:
                                    self.feature_vectors_dict[word]['cool'][0] += lineCool * vectorizer.vocabulary_[
                                        word] / max(total_words, 1.0)
                                    self.feature_vectors_dict[word]['cool'][1] += vectorizer.vocabulary_[word]

                    # Handling for text that is all stopwords, such as "a a a a a" or "x"
                    except ValueError:
                        null_samples += 1

                # Else handle misshapen lines
                else:
                    null_samples += 1

                total_samples += 1

            self.total_samples = total_samples
            time_taken = time.time() - start

            print('Completed chunk up to {} in {:.2f} sec. \nFound {} null entries in chunk'
                  .format(total_samples, time_taken, total_null_samples))

        # NORMALIZATION OF PROBABILITIES

        train_time = datetime.datetime.now() - train_start

        print('Completed training on {} items in {}.'.format(total_samples, train_time))
        print('Normalizing probability matrix...')
        normal_start = datetime.datetime.now()

        # CUSTOM REDUCING NORMALIZER
        if USE_CUSTOM_NORMALIZER:
            for rating in self.p_has_word_category.values('stars'):
                for word in self.p_has_word_category.values('hasWord'):
                    self.p_has_word_category[word, rating] = self.p_has_word_category[word, rating] / float(
                        word_totals[rating])

        # LAB NAIVE BAYES NORMALIZER
        else:
            # Iterate over ratings and normalize probabilities.
            for rating in self.p_has_word_category.values('stars'):
                full_count = sum(
                    (self.p_has_word_category[word, rating] for word in self.p_has_word_category.values('hasWord')))
                for word in self.p_has_word_category.values('hasWord'):
                    self.p_has_word_category[word, rating] = self.p_has_word_category[word, rating] / float(full_count)

        normal_time = datetime.datetime.now() - normal_start
        print('Normalization completed in {}.'.format(normal_time))
        print('Done Training.')

    def save_model(self):
        """Funtion to save the currently loaded model as a pickle file. Saves to location in self.outputFile"""
        print('Saving model...')

        try:
            f = open(self.modelFile, 'wb')
            pickle.dump(self, f)
            f.close()
        except IOError as e:
            print('Failed to save model to {}.\n{}'.format(self.modelFile, e))
            sys.exit(0)

        print('Model saved to {}.'.format(self.modelFile))

    def load_model(self):
        """Function to load a pickle file as model. Loads from location in self.modelFile. Will attempt to load any
        pickle file as the model. Loading unrecognized models will cause the program to fail."""
        print('Loading model from {}...'.format(self.modelFile))

        try:
            f = open(self.modelFile, 'rb')
            model = pickle.load(f)

            """ # bugs out running from parent file, but not this file... what?
            if not isinstance(model, NaiveBayes):
                print('Failed to load model as Naive Bayes Classifier. File may be malformed, or not created by '
                      'NaiveBayes.save_model().')
                print('Program terminated.')
                sys.exit(0)
            """

            self.feature_vectors_dict = model.feature_vectors_dict
            self.p_has_word_category = model.p_has_word_category
            self.p_stars = model.p_stars
            self.total_samples = model.total_samples
            self.incFeatures = model.incFeatures
            f.close()

        except IOError as e:
            print('Failed to load model from {}.\n{}'.format(self.modelFile, e))
            f.close()
            sys.exit(0)

        print('Model loaded.')

    def run_test(self):
        """Function to test loaded model against test data. Uses model loaded into memory and test data from
        self.inputFile. Prints results to console directly."""

        test_total = 0
        test_correct = 0
        test_incorrect = 0
        test_null = 0
        useful_residual_absolute_total = 0
        useful_residual_squared_total = 0
        funny_residual_absolute_total = 0
        funny_residual_squared_total = 0
        cool_residual_absolute_total = 0
        cool_residual_squared_total = 0

        zero_prob = 1.0 / float(self.total_samples)

        print('Running predictions on test file at ' + self.testFile + ' ...')
        test_start = datetime.datetime.now()

        # Load chunk from testfile with pandas
        df = pd.read_json(self.testFile, lines=True, chunksize=TESTING_CHUNK_SIZE)

        # For each chunk of test data, categorize and compare results
        for chunk in df:

            # Training early stop for faster testing
            if test_total >= MAX_TESTING:
                print("Exceeded maximum number of samples.")
                break

            print('Testing with chunk of ' + str(chunk.shape[0]) + ' samples...')
            chunk_start = time.time()

            # For Each line in chunk, run classification and regression
            for line_index, line in chunk.iterrows():

                # should've normalized this as part of the model, but I already pickled and don't want to retrain.
                line_prob = dict({
                    1: float(self.p_stars[1]) / float(self.total_samples),
                    2: float(self.p_stars[2]) / float(self.total_samples),
                    3: float(self.p_stars[3]) / float(self.total_samples),
                    4: float(self.p_stars[4]) / float(self.total_samples),
                    5: float(self.p_stars[5]) / float(self.total_samples),
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
                    try:
                        # Count word occurrences in line
                        vectorizer = CountVectorizer(analyzer='word', stop_words='english')  #
                        vectorizer.fit([line['text']])

                        # For each word in line, add line probability for the number of occurrences
                        for word in vectorizer.vocabulary_:

                            occurences = vectorizer.vocabulary_[word]

                            # For rating category
                            for cat in range(1, 6):

                                # find P (has word | category)
                                prob = self.p_has_word_category[word, cat]

                                # if word prob is zero for category, set to near zero value instead
                                if prob == 0:
                                    prob = zero_prob

                                # Add P (has word | category) for number of occurrences of word
                                line_prob[cat] += prob * occurences

                            # if feature regression, add word vector
                            if self.incFeatures:
                                feat = self.get_feature_vector(word)
                                line_useful += float(feat['useful'])
                                line_funny += float(feat['funny'])
                                line_cool += float(feat['cool'])

                        # Find highest prob classification for line and compare to actual
                        rating = max(line_prob, key=line_prob.get)
                        if rating is line_rating_ac:
                            test_correct += 1
                        else:
                            test_incorrect += 1

                        # If features enabled, check regression error, add to totals.
                        if self.incFeatures:
                            residual_absolute = (abs(line_useful - line_useful_ac) + abs(line_funny - line_funny_ac)
                                                 + abs(line_cool - line_cool_ac))

                            useful_residual_absolute_total += abs(line_useful - line_useful_ac)
                            funny_residual_absolute_total += abs(line_funny - line_funny_ac)
                            cool_residual_absolute_total += abs(line_cool - line_cool_ac)

                            useful_residual_squared_total += numpy.square(line_useful - line_useful_ac)
                            funny_residual_squared_total += numpy.square(line_funny - line_funny_ac)
                            cool_residual_squared_total += numpy.square(line_cool - line_cool_ac)

                    # Handling for text that is all stopwords, such as "a a a a a" or "x"
                    except ValueError as e:
                        test_null += 1

                # Increment test total for completed line
                test_total += 1

            chunk_time = time.time() - chunk_start
            print('Completed testing chunk up to {} samples in {:.2f} sec.'.format(test_total, chunk_time))

        # Calculate test statistics
        test_time = datetime.datetime.now() - test_start

        accuracy = test_correct / test_total * 100.00

        print('Completed prediction on {} samples in {}.'.format(test_total, test_time))
        print('Found {} null entries in test file.'.format(test_null))
        print('Rating Accuracy:     {:.2f}%'.format(accuracy))

        if self.incFeatures:
            useful_mae = useful_residual_absolute_total / test_total
            useful_rmse = numpy.sqrt(useful_residual_squared_total / test_total)
            cool_mae = cool_residual_absolute_total / test_total
            cool_rmse = numpy.sqrt(cool_residual_squared_total / test_total)
            funny_mae = funny_residual_absolute_total / test_total
            funny_rmse = numpy.sqrt(funny_residual_squared_total / test_total)
            avg_mae = numpy.average([useful_mae, cool_mae, funny_mae])
            avg_rmse = numpy.average([useful_rmse, funny_rmse, cool_rmse])

            print('Avg MAE:             {:.4f}'.format(avg_mae))
            print('Avg RMSE:            {:.4f}'.format(avg_rmse))
            print('Useful MAE:          {:.4f}'.format(useful_mae))
            print('Useful RMSE:         {:.4f}'.format(useful_rmse))
            print('Funny MAE:           {:.4f}'.format(funny_mae))
            print('Funny RMSE:          {:.4f}'.format(funny_rmse))
            print('Cool MAE:            {:.4f}'.format(cool_mae))
            print('Cool RMSE:           {:.4f}'.format(cool_rmse))


def main():
    NaiveBayes(parse_args(argparse.ArgumentParser(description='Probabalistic classifier')))


if __name__ == "__main__":
    main()
