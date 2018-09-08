import os
import numpy as np
import math
import operator

from collections import Counter
from os import listdir

from os.path import isfile, join
from scipy import spatial
from nltk import RegexpTokenizer
from nltk.corpus import stopwords


def read_input(input_data):
    """
    The following function is used to read the inputs for text processing from data file
    :return:
    """
    with open("data/" + input_data, encoding="utf8") as file:
        lines = file.readlines()
        lines = [line.strip() for line in lines]
    return lines


def find_line_unigrams_bow(lines):
    """
    Using that function we transform lines from sentences to words.
    :param lines: lines
    :return: set of words per line
    """
    tokenizer = RegexpTokenizer(r'\w+')
    stop_words = set(stopwords.words('english'))
    unigrams = []
    bows = []
    find_features = []
    for line in lines:
        line_unigrams = [word.lower() for word in tokenizer.tokenize(line) if word not in stop_words]
        if line_unigrams == []:
            continue
        else:
            unigrams.append(line_unigrams)
            line_bow = Counter(line_unigrams)
            find_features += list(line_bow.keys())
            bows.append(line_bow)
    features = list(set(find_features))
    return unigrams, bows, features


def build_feature_matrix(list_of_bags, features):
    """
    Using that function you are able to build the feature matrix.
    :param list_of_bags: list of text bows
    :return: feature matrix
    """
    feature_matrix = np.zeros((len(list_of_bags), len(features)))
    for i in range(0, feature_matrix.shape[0]):
        current_bow = list_of_bags[i]
        for j in range(0, feature_matrix.shape[1]):
            if features[j] in current_bow.keys():
                feature_matrix[i, j] = current_bow[features[j]]
            else:
                feature_matrix[i, j] = 0
    return feature_matrix


def calculate_tfidf(feature_matrix):
    """
    Using that function we are able to calculate the tfidf feature matrix.
    :param feature_matrix: tf feature matrix
    :return: tfidf feature matrix
    """
    for j in range(0, feature_matrix.shape[1]):
        term_df = np.count_nonzero(feature_matrix[:, j])
        for i in range(0, feature_matrix.shape[0]):
            temp_eq = math.log((1 + feature_matrix.shape[0]) / (1 + term_df)) + 1
            feature_matrix[i, j] = feature_matrix[i, j] * temp_eq
    return feature_matrix


def _similarity(tfidif_matrix):
    """
    The followign matrix calculates the
    :param tfidif_matrix:
    :return:
    """
    for i in range(0, tfidif_matrix.shape[0]):
        similarity = {}
        for j in range(0, tfidif_matrix.shape[0]):
            if i == j:
                continue
            else:
                similarity[j] = 1 - spatial.distance.cosine(tfidif_matrix[i, :], tfidif_matrix[j, :])
        print("The most similar text for line {} is {} with cosine similarity calculated to {}".format(i, max(
            similarity.items(), key=operator.itemgetter(1))[0], similarity[max(similarity.items(),
                                                                               key=operator.itemgetter(1))[0]]))
