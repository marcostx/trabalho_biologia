import collections, sys
import itertools
from parser import load_csv
import numpy as np
import argparse
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Embedding
from keras.constraints import maxnorm
from keras.optimizers import SGD
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
import itertools
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from keras.utils import np_utils
from keras import initializers
from matplotlib import pyplot
from keras.utils.np_utils import to_categorical
import keras
from utils import *
# from seya.layers.recurrent import Bidirectional




def create_baseline_model(num_classes, inp_shape):
    
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(1, 1), activation='relu', input_shape=(inp_shape[1], inp_shape[0], inp_shape[2])))
    model.add(MaxPooling2D(pool_size=(1, 1)))
    model.add(Conv2D(32, kernel_size=(2, 2), activation='relu'))
    model.add(MaxPooling2D(pool_size=(1, 1)))
    model.add(Flatten())
    model.add(Dense(100, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))
    epochs = 30
    lrate = 0.001
    decay = lrate / epochs
    sgd = Adam(lr=lrate, epsilon=1e-08, decay=decay)
    model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])

    return model, epochs


def cross_dataset_train(X,y,batch_size):

    y=to_categorical(y)
    
    model, epochs = create_baseline_model(y.shape[1],X.shape[1:])
    X = X.reshape((X.shape[0],X.shape[2],X.shape[1],X.shape[3]))

    model.fit(X, y, batch_size=batch_size, epochs=epochs, shuffle=True,verbose=False)

    return model


def cross_dataset_evaluation(model,X_b,y_b):
    y_b=to_categorical(y_b)
    X_b = X_b.reshape((X_b.shape[0],X_b.shape[2],X_b.shape[1],X_b.shape[3]))
    pred = model.predict(X_b, verbose=0)

    pred = [np.argmax(item) for item in pred]
    y_b = [np.argmax(item) for item in y_b]

    print("CROSS-DATASET EVALUATION: ")
    print("accuracy : ", accuracy_score(y_b, pred))
    print("precision : ", precision_score(y_b, pred, average='weighted'))
    print("recall : ", recall_score(y_b, pred, average='weighted'))
    print("f1 : ", f1_score(y_b, pred, average='weighted'))
    print("\n")


if __name__ == '__main__':
    datasets=["H3-clean.csv","H4-clean.csv","H3K4me1-clean.csv",
        "H3K4me2-clean.csv","H3K4me3-clean.csv","H3K9ac-clean.csv",
        "H3K14ac-clean.csv","H3K36me3-clean.csv","H3K79me3-clean.csv",
        "H4ac-clean.csv"]

    # assert len(sys.argv[1]) > 1
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_dataset', help='input file', required=True)
    parser.add_argument('-c', '--cross_dataset', help='cross dataset evaluation', required=False)

    ARGS = parser.parse_args()
    X, y = load_csv(ARGS.input_dataset)

    X = preprocess(X)
    inp_shape = X.shape

    if ARGS.cross_dataset:
        print("cross dataset experiment")
        

        model = cross_dataset_train(X,y,128)

        for dataset in datasets:
            if dataset == ARGS.input_dataset:
                continue
            print("evaluating : ",dataset)
            X_b,y_b=load_csv(dataset)
            X_b = X = preprocess(X_b)

            cross_dataset_evaluation(model,X_b,y_b)
            
        exit()

    #X_train, X_test, y_train, y_test = train_test_split(X_binarized, y, test_size=0.33, random_state=42)

    #y_train = np_utils.to_categorical(y_train)
    #y_test_ = np_utils.to_categorical(y_test)

    #num_classes = y_train.shape[1]

    #model, epochs = create_baseline_model(num_classes, inp_shape)

    #seed = 7
    #np.random.seed(seed)

    #model.fit(X_train, y_train, validation_data=(X_test, y_test_), nb_epoch=epochs, batch_size=50)
    # Final evaluation of the model
    #pred = model.predict(X_test, verbose=0)
    #pred = [np.argmax(item) for item in pred]

    #print("accuracy : ", accuracy_score(y_test, pred))
    #print("precision : ", precision_score(y_test, pred, average='weighted'))
    #print("recall : ", recall_score(y_test, pred, average='weighted'))
    #print("f1 : ", f1_score(y_test, pred, average='weighted'))
    #print("\n")
