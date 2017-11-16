import collections, sys
from Bio import Seq, SeqIO, SeqRecord
import itertools
from parser import load_csv
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.constraints import maxnorm
from keras.optimizers import SGD
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
import itertools
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from keras.utils import np_utils
from keras.layers.recurrent import LSTM
from keras.layers import SimpleRNN
from keras import initializers


def preprocess(X):
    X_binarized = [binary_representation(item) for item in X]
    X_binarized = np.array(X_binarized)

    # X_binarized = X_binarized.reshape((X_binarized.shape[0],X_binarized.shape[1]))

    return X_binarized


def create_dict_words():
    kmer = 3
    nucleotides = ['a', 't', 'g', 'c']
    words = []
    binary_words = {}
    combinations = itertools.product(*itertools.repeat(nucleotides, 3))
    for idx, j in enumerate(combinations):
        words.append(''.join(j))
        _ = np.zeros((len(nucleotides) * len(nucleotides) * len(nucleotides)))
        _[idx] = 1.0
        binary_words[''.join(j)] = _

    return binary_words


def twin(km):
    return Seq.reverse_complement(km)


def kmers(seq, k):
    kmer = []
    for i in range(len(seq) - k + 1):
        kmer.append(seq[i:i + k])

    return kmer


def binary_representation(fn, k=3, limit=1):
    d = collections.defaultdict(int)
    dict_words = create_dict_words()

    seq_l = fn
    kms = kmers(seq_l, k)

    representation = []
    for idx in range(0, len(kms) - 1):
        f_ = np.array(dict_words[kms[idx]])
        s_ = np.array(dict_words[kms[idx + 1]])

        representation.append(np.append(f_, s_))

    representation = np.array(representation).flatten()

    return np.array(representation)

def get_words(vector_sentences):
    sentences = []
    k = 3

    for sentence in vector_sentences:
        words = []
        for i in range(0, len(sentence) - k + 1, k):
            words.append(sentence[i:i + k])

        sentences.append(words)

    return sentences


def transform_to_vectors(X, dict_words):
    new_x = []
    k = 3

    for sentence in X:
        words = [dict_words[sentence[i:i + k]] for i in range(0, len(sentence) - k + 1, k)]
        new_x.append(np.array(words).flatten())

    return np.array(new_x)

def get_binary_words(vector_sentences):
    sentences = []
    nucleotides={'a':[1,0,0,0],'t':[0,1,0,0],'c':[0,0,1,0],'g':[0,0,0,1]}
    for sentence in vector_sentences:
        words = []
        for i in range(0, len(sentence)):
            words.append(nucleotides[sentence[i]])

        sentences.append(words)
    sentences = np.array(sentences)

    return sentences
