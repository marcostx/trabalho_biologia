import collections, sys
from parser import load_csv
import numpy as np
import argparse
import itertools
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from keras.utils import np_utils
from sklearn.model_selection import train_test_split
from keras import initializers
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt
from utils import *
import warnings
from sklearn.svm import SVC
import collections, sys
from parser import load_csv
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
from sklearn.model_selection import LeaveOneOut,StratifiedKFold
import matplotlib.pyplot as plt
from utils import *
import warnings
warnings.filterwarnings('ignore')

def create_model(num_classes,inp_shape,simple=False):
    epochs = 5

    print('building model')

    model = Sequential()
    model.add(Dense(100, activation='relu', input_shape=inp_shape))
    model.add(Dense(num_classes, activation='softmax'))

    print('compiling model')
    if num_classes==2:
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    else:
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model, epochs


def train_and_evaluate(X,y,batch_size,splits,simple=False):
    #X_t, X_val, y_t, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # validation split

    fold=0

    kf = StratifiedKFold(n_splits=splits)
    accs,pres,recalls,f1s = [],[],[],[]

    for train_index, test_index in kf.split(X,y):

        print("Fold : ", fold)

        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
       
        y_train = to_categorical(y_train)
        y_test  = to_categorical(y_test)

        model, epochs = create_model(y_train.shape[1],X_train.shape[1:])
        model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, shuffle=True,verbose=False,validation_data=(X_test,y_test))

        pred = model.predict(X_test, verbose=0)

        pred = [np.argmax(item) for item in pred]
        y_test = [np.argmax(item) for item in y_test]
        print("accuracy : ", accuracy_score(y_test, pred))
        #print("precision : ", precision_score(y_test, pred, average='weighted'))
        #print("recall : ", recall_score(y_test, pred, average='weighted'))
        #print("f1 : ", f1_score(y_test, pred, average='weighted'))
        #print("\n")
        accs.append(accuracy_score(y_test, pred))
        pres.append(precision_score(y_test, pred, average='weighted'))
        recalls.append(recall_score(y_test, pred, average='weighted'))
        f1s.append(f1_score(y_test, pred, average='weighted'))
        fold+=1


        del model

    results = [accs,pres,recalls,f1s]

    return results


if __name__ == '__main__':
    datasets=["H3-clean.csv","H4-clean.csv","H3K4me1-clean.csv",
        "H3K4me2-clean.csv","H3K4me3-clean.csv","H3K9ac-clean.csv",
        "H3K14ac-clean.csv","H3K36me3-clean.csv","H3K79me3-clean.csv",
        "H4ac-clean.csv"]


    #parser = argparse.ArgumentParser()
    #parser.add_argument('-i', '--input_dataset', help='input file', required=False)
    splits=10
    #ARGS = parser.parse_args()
    for dataset in datasets:
        print("Evaluating , ",dataset)
        X,y = load_csv(dataset)

        X = get_binary_words(X)

        results = train_and_evaluate(X,y,128,splits)

        print("mean metrics cv=10")
        print("accuracy : mean={}, std={}".format(np.mean(results[0]),np.std(results[0])))
        print("precision : mean={}, std={}".format(np.mean(results[1]),np.std(results[1])))
        print("recall : mean={}, std={}".format(np.mean(results[2]),np.std(results[2])))
        print("f1 : mean={}, std={}".format(np.mean(results[3]),np.std(results[3])))
        print("\n")
