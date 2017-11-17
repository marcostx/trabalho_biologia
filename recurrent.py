import collections, sys
from parser import load_csv
import numpy as np
import argparse
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Embedding
from keras.constraints import maxnorm
from keras.optimizers import SGD
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import Adam
import itertools
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from keras.utils import np_utils
from keras.layers.recurrent import LSTM
from keras.layers import SimpleRNN
from sklearn.feature_extraction.text import CountVectorizer
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical
from keras import initializers
from sklearn.model_selection import StratifiedKFold
from utils import *
from keras.layers.convolutional import Conv1D, MaxPooling1D
from keras.layers import LSTM, Bidirectional
from utils import *


def create_recurrent_model(num_classes,inp_shape):
    epochs = 30

    print 'building model'

    model = Sequential()
    model.add(Conv1D(activation="relu", input_shape=inp_shape, padding="valid", strides=1, filters=200, kernel_size=3))

    model.add(MaxPooling1D(strides=2, pool_size=2))
    model.add(Dropout(0.2))
    model.add(Bidirectional(LSTM(128, return_sequences=True)))

    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(300, activation='relu'))

    model.add(Dense(num_classes, activation='softmax'))

    print 'compiling model'
    model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

    return model, epochs


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_dataset', help='input file', required=True)

    ARGS = parser.parse_args()

    X, y = load_csv(ARGS.input_dataset)

    #X = get_binary_words(X)
    X = get_binary_words(X)

    
    fold=0

    #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    skf = StratifiedKFold(n_splits=10)
    accs,pres,recalls,f1s = [],[],[],[]

    for train_index, test_index in skf.split(X, y):
        print("Fold : ", fold)
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        y_train = to_categorical(y_train)
        y_test  = to_categorical(y_test)

        model, epochs = create_recurrent_model(y_train.shape[1],X_train.shape[1:])

        model.fit(X_train, y_train, batch_size=128, epochs=epochs, shuffle=True,verbose=False,validation_data=(X_test, y_test))

        pred = model.predict(X_test, verbose=0)

        pred = [np.argmax(item) for item in pred]
        y_test = [np.argmax(item) for item in y_test]


        print("accuracy : ", accuracy_score(y_test, pred))
        print("precision : ", precision_score(y_test, pred, average='weighted'))
        print("recall : ", recall_score(y_test, pred, average='weighted'))
        print("f1 : ", f1_score(y_test, pred, average='weighted'))
        print("\n")
        accs.append(accuracy_score(y_test, pred))
        pres.append(precision_score(y_test, pred, average='weighted'))
        recalls.append(recall_score(y_test, pred, average='weighted'))
        f1s.append(f1_score(y_test, pred, average='weighted'))
        fold+=1

    print("mean metrics cv=10")
    print("accuracy : ", np.mean(accs))
    print("precision : ",np.mean(pres))
    print("recall : ", np.mean(recalls))
    print("f1 : ", np.mean(f1s))
    print("\n")