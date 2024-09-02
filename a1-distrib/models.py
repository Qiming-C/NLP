# models.py
import random
from collections import Counter

import nltk
import numpy as np

from sentiment_data import *
from utils import *

# nltk.download('stopwords')

class FeatureExtractor(object):
    """
    Feature extraction base type. Takes a sentence and returns an indexed list of features.
    """

    def get_indexer(self):
        raise Exception("Don't call me, call my subclasses")

    def extract_features(self, sentence: List[str], add_to_indexer: bool = False) -> Counter:
        """
        Extract features from a sentence represented as a list of words. Includes a flag add_to_indexer to
        :param sentence: words in the example to featurize
        :param add_to_indexer: True if we should grow the dimensionality of the featurizer if new features are encountered.
        At test time, any unseen features should be discarded, but at train time, we probably want to keep growing it.
        :return: A feature vector. We suggest using a Counter[int], which can encode a sparse feature vector (only
        a few indices have nonzero value) in essentially the same way as a map. However, you can use whatever data
        structure you prefer, since this does not interact with the framework code.
        """
        raise Exception("Don't call me, call my subclasses")


class UnigramFeatureExtractor(FeatureExtractor):
    """
    Extracts unigram bag-of-words features from a sentence. It's up to you to decide how you want to handle counts
    and any additional preprocessing you want to do.
    """

    def __init__(self, indexer: Indexer):
        self.indexer = indexer

    def get_indexer(self):
        return self.indexer

    def extract_features(self, sentence: List[str], add_to_indexer: bool = False) -> Counter:

        feature_vector = Counter()

        for word in sentence:
            word = word.lower()
            # if word not in self.stopwords:
            ## finding: stopwords make the accuracy worse
            if add_to_indexer:
                index = self.indexer.add_and_get_index(word)
            else:
                index = self.indexer.index_of(word)
                if index == -1:
                    continue
            feature_vector[index] += 1

        return feature_vector


class BigramFeatureExtractor(FeatureExtractor):
    """
    Bigram feature extractor analogous to the unigram one.
    """

    def __init__(self, indexer: Indexer):
        raise Exception("Must be implemented")


class BetterFeatureExtractor(FeatureExtractor):
    """
    Better feature extractor...try whatever you can think of!
    """

    def __init__(self, indexer: Indexer):
        raise Exception("Must be implemented")


class SentimentClassifier(object):
    """
    Sentiment classifier base type
    """

    def predict(self, sentence: List[str]) -> int:
        """
        :param sentence: words (List[str]) in the sentence to classify
        :return: Either 0 for negative class or 1 for positive class
        """
        raise Exception("Don't call me, call my subclasses")


class TrivialSentimentClassifier(SentimentClassifier):
    """
    Sentiment classifier that always predict    s the positive class.
    """

    def predict(self, sentence: List[str]) -> int:
        return 1


class PerceptronClassifier(SentimentClassifier):
    """
    Implement this class -- you should at least have init() and implement the predict method from the SentimentClassifier
    superclass. Hint: you'll probably need this class to wrap both the weight vector and featurizer -- feel free to
    modify the constructor to pass these in.
    """

    def __init__(self, featureExtractor: FeatureExtractor):
        self.features = None
        self.featureExtractor = featureExtractor
        self.weights = np.zeros(self.featureExtractor.get_indexer().__len__())

    def predict(self, sentence: List[str]) -> int:
        self.features = self.featureExtractor.extract_features(sentence)

        ## compute the W
        w = sum(self.weights[index] * value for index, value in self.features.items())

        return 1 if w > 0 else 0

    def update_weight(self, sentence: List[str], label: int, learning_rate: float = 1.15):
        ## get the pred
        pred = self.predict(sentence)

        error = label - pred

        if error != 0:
            for index, value in self.features.items():
                self.weights[index] += learning_rate * (error / 2) * value


class LogisticRegressionClassifier(SentimentClassifier):
    """
    Implement this class -- you should at least have init() and implement the predict method from the SentimentClassifier
    superclass. Hint: you'll probably need this class to wrap both the weight vector and featurizer -- feel free to
    modify the constructor to pass these in.
    """

    def __init__(self, featureExtractor: FeatureExtractor):
        self.features = None
        self.featureExtractor = featureExtractor
        self.weights = np.zeros(self.featureExtractor.get_indexer().__len__())
        self.decay_rate = 0.256
        self.initial_learning_rate = 0.985


    def sigmoid(self, w):
        return 1 / (1 + np.exp(-w))

    def predict(self, sentence: List[str]) -> int:
        self.features = self.featureExtractor.extract_features(sentence)

        w = sum(self.weights[index] * value for index, value in self.features.items())


        return 1 if self.sigmoid(w) > 0.5 else 0

    def update_weight(self, sentence: List[str], label: int, epoch: int):
        ## exp decay learning rate
        learning_rate = self.initial_learning_rate * np.exp(-self.decay_rate * epoch)

        pred = self.predict(sentence)

        error = label - pred

        for index, value in self.features.items():
            self.weights[index] += learning_rate * error * value





def train_perceptron(train_exs: List[SentimentExample], feat_extractor: FeatureExtractor) -> PerceptronClassifier:
    """
    Train a classifier with the perceptron.
    :param train_exs: training set, List of SentimentExample objects
    :param feat_extractor: feature extractor to use
    :return: trained PerceptronClassifier model
    """

    for sentence in train_exs:
        feat_extractor.extract_features(sentence.words, add_to_indexer=True)

    model = PerceptronClassifier(feat_extractor)

    epochs = 30
    for epoch in range(epochs):
        # shuffle the data
        random.shuffle(train_exs)

        for data in train_exs:
            sentence = data.words
            label = data.label
            model.update_weight(sentence, label)

    return model


def train_logistic_regression(train_exs: List[SentimentExample],
                              feat_extractor: FeatureExtractor) -> LogisticRegressionClassifier:
    """
    Train a logistic regression model.
    :param train_exs: training set, List of SentimentExample objects
    :param feat_extractor: feature extractor to use
    :return: trained LogisticRegressionClassifier model
    """
    for sentence in train_exs:
        feat_extractor.extract_features(sentence.words, add_to_indexer=True)

    model = LogisticRegressionClassifier(feat_extractor)

    epochs = 30
    for epoch in range(epochs):
        # shuffle the data
        random.shuffle(train_exs)

        for data in train_exs:
            sentence = data.words
            label = data.label
            model.update_weight(sentence, label, epoch)

    return model


def train_model(args, train_exs: List[SentimentExample], dev_exs: List[SentimentExample]) -> SentimentClassifier:
    """
    Main entry point for your modifications. Trains and returns one of several models depending on the args
    passed in from the main method. You may modify this function, but probably will not need to.
    :param args: args bundle from sentiment_classifier.py
    :param train_exs: training set, List of SentimentExample objects
    :param dev_exs: dev set, List of SentimentExample objects. You can use this for validation throughout the training
    process, but you should *not* directly train on this data.
    :return: trained SentimentClassifier model, of whichever type is specified
    """
    # Initialize feature extractor
    if args.model == "TRIVIAL":
        feat_extractor = None
    elif args.feats == "UNIGRAM":
        # Add additional preprocessing code here
        feat_extractor = UnigramFeatureExtractor(Indexer())
    elif args.feats == "BIGRAM":
        # Add additional preprocessing code here
        feat_extractor = BigramFeatureExtractor(Indexer())
    elif args.feats == "BETTER":
        # Add additional preprocessing code here
        feat_extractor = BetterFeatureExtractor(Indexer())
    else:
        raise Exception("Pass in UNIGRAM, BIGRAM, or BETTER to run the appropriate system")

    # Train the model
    if args.model == "TRIVIAL":
        model = TrivialSentimentClassifier()
    elif args.model == "PERCEPTRON":
        model = train_perceptron(train_exs, feat_extractor)
    elif args.model == "LR":
        model = train_logistic_regression(train_exs, feat_extractor)
    else:
        raise Exception("Pass in TRIVIAL, PERCEPTRON, or LR to run the appropriate system")
    return model
