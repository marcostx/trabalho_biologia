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
from keras.layers.normalization import BatchNormalization
from keras.utils.np_utils import to_categorical
from keras import initializers
from sklearn.model_selection import LeaveOneOut,KFold
from keras.layers.convolutional import Conv1D, MaxPooling1D
from keras.layers import LSTM, Bidirectional
import matplotlib.pyplot as plt
from utils import *

def create_recurrent_model(num_classes,inp_shape):
    epochs = 5

    print('building model')

    model = Sequential()
    model.add(Conv1D(activation="relu", input_shape=inp_shape, padding="same", strides=1, filters=500, kernel_size=3))
    model.add(MaxPooling1D(strides=2, pool_size=2))
    model.add(Dropout(0.2))
    model.add(Conv1D(activation="relu", padding="same", strides=1, filters=300, kernel_size=3))
    model.add(MaxPooling1D(strides=2, pool_size=2))
    model.add(Dropout(0.2))
    model.add(Bidirectional(LSTM(128, return_sequences=True)))

    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(300, activation='relu'))

    model.add(Dense(num_classes, activation='softmax'))

    print('compiling model')
    if num_classes==2:
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    else:
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model, epochs

def train_and_evaluate(X,y,batch_size,splits):
    
    fold=0
    kf = KFold(n_splits=splits)
    accs,pres,recalls,f1s = [],[],[],[]

    for train_index, test_index in kf.split(X):

        print("Fold : ", fold)
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        y_train = to_categorical(y_train)
        y_test  = to_categorical(y_test)
        print(X_train.shape, y_train.shape)

        model, epochs = create_recurrent_model(y_train.shape[1],X_train.shape[1:])

        model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, shuffle=True,verbose=True,validation_data=(X_test,y_test))

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
        plt.figure()
        cnf_matrix = confusion_matrix(y_test, pred)
        plt.save_fig("confusion_matrix_" + str(fold) + ".png")
        

        del model
    results = [acc,pres,recalls,f1s]

    return results

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_dataset', help='input file', required=True)
    parser.add_argument('-b', '--batch_size', help='input file', required=False)

    ARGS = parser.parse_args()

    X,y = load_csv(ARGS.input_dataset)
    if ARGS.batch_size:
        batch_size=ARGS.batch_size
    else:
        batch_size=256
    
    X = get_binary_words(X)

    splits=10
    results = train_and_evaluate(X,y,batch_size,splits)
    
    print("mean metrics cv=10")
    print("accuracy : mean={}, std={}".format(np.mean(results[0]),np.std(results[0])))
    print("precision : mean={}, std={}".format(np.mean(results[1]),np.std(results[1])))
    print("recall : mean={}, std={}".format(np.mean(results[2]),np.std(results[2])))
    print("f1 : mean={}, std={}".format(np.mean(results[3]),np.std(results[3])))
    print("\n")



    """
    leave one out :
    mean metrics cv=10
    ('accuracy : ', 0.78301886792452835)
    ('precision : ', 0.78301886792452835)
    ('recall : ', 0.78301886792452835)
    ('f1 : ', 0.78301886792452835)
    """
