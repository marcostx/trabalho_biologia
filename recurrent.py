import collections, sys
from parser import load_csv
import numpy as np
import argparse
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from keras.layers import Dense, LSTM
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical
from sklearn.model_selection import StratifiedKFold
from keras.layers.convolutional import Conv1D, MaxPooling1D
from keras.layers import LSTM, Bidirectional
from utils import *
import warnings
warnings.filterwarnings('ignore')

def create_recurrent_model(num_classes,inp_shape,simple=False):
    epochs = 120

    print('building model')

    model = Sequential()
    model.add(Conv1D(activation="relu", input_shape=inp_shape, padding="same", strides=1, filters=500, kernel_size=3))
    model.add(MaxPooling1D(strides=2, pool_size=2))
    model.add(Dropout(0.2))
    model.add(Conv1D(activation="relu", padding="same", strides=1, filters=300, kernel_size=3))
    model.add(MaxPooling1D(strides=2, pool_size=2))
    model.add(Dropout(0.2))
    if simple:
        model.add(LSTM(128, return_sequences=True))
    else:
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

def train_and_evaluate(X,y,batch_size,splits,simple=False):
    fold=0

    kf = StratifiedKFold(n_splits=splits)
    accs,pres,recalls,f1s = [],[],[],[]

    for train_index, test_index in kf.split(X,y):

        print("Fold : ", fold)

        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]


        y_train = to_categorical(y_train)
        y_test  = to_categorical(y_test)


        if simple:
            model, epochs = create_recurrent_model(y_train.shape[1],X_train.shape[1:],simple=True)
        else:            
            model, epochs = create_recurrent_model(y_train.shape[1],X_train.shape[1:])

        model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs,verbose=True,validation_data=(X_test,y_test))

        pred = model.predict(X_test, verbose=0)

        pred = [np.argmax(item) for item in pred]
        y_test = [np.argmax(item) for item in y_test]


        #print("accuracy : ", accuracy_score(y_test, pred))
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

def train_and_evaluate_hold_out(X,y,batch_size,splits):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # validation split

    fold=0

    y_train = to_categorical(y_train)
    y_test  = to_categorical(y_test)

    model, epochs = create_recurrent_model(y_train.shape[1],X_train.shape[1:])

    model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs,verbose=True,validation_data=(X_test,y_test))

    pred = model.predict(X_test, verbose=0)

    pred = [np.argmax(item) for item in pred]
    y_test = [np.argmax(item) for item in y_test]

    

    print("accuracy : ", accuracy_score(y_test, pred))
    print("precision : ", precision_score(y_test, pred, average='weighted'))
    print("recall : ", recall_score(y_test, pred, average='weighted'))
    print("f1 : ", f1_score(y_test, pred, average='weighted'))
    print("\n")


def train_and_evaluate_validation(X,y,batch_size,splits,simple=False):
    X_t, X_test, y_t, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    fold=0

    kf = StratifiedKFold(n_splits=splits)
    accs,pres,recalls,f1s = [],[],[],[]

    for train_index, test_index in kf.split(X_t,y_t):

        print("Fold : ", fold)

        X_train, X_val = X[train_index], X[test_index]
        y_train, y_val = y[train_index], y[test_index]

        y_train = to_categorical(y_train)
        y_val  = to_categorical(y_val)

        model, epochs = create_recurrent_model(y_train.shape[1],X_train.shape[1:])
        model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs,verbose=False,validation_data=(X_val,y_val))

        pred = model.predict(X_val, verbose=0)

        pred = [np.argmax(item) for item in pred]
        y_val = [np.argmax(item) for item in y_val]
        #print("accuracy : ", accuracy_score(y_val, pred))

        accs.append(accuracy_score(y_val, pred))
        pres.append(precision_score(y_val, pred, average='weighted'))
        recalls.append(recall_score(y_val, pred, average='weighted'))
        f1s.append(f1_score(y_val, pred, average='weighted'))
        fold+=1


        del model

    results = [accs,pres,recalls,f1s]

    return results


def cross_dataset_train(X,y,batch_size):

    y=to_categorical(y)

    model, epochs = create_recurrent_model(y.shape[1],X.shape[1:])

    model.fit(X, y, batch_size=batch_size, epochs=epochs, shuffle=True,verbose=False)

    return model


def cross_dataset_evaluation(model,X_b,y_b):
    y_b=to_categorical(y_b)

    pred = model.predict(X_b, verbose=0)

    pred = [np.argmax(item) for item in pred]
    y_b = [np.argmax(item) for item in y_b]

    print("CROSS-DATASET EVALUATION: ")
    print("accuracy : ", accuracy_score(y_b, pred))
    print("precision : ", precision_score(y_b, pred, average='weighted'))
    print("recall : ", recall_score(y_b, pred, average='weighted'))
    print("f1 : ", f1_score(y_b, pred, average='weighted'))
    print("\n")

    results_cross = open('results_cross.txt','a')
    txt=["CROSS-DATASET EVALUATION: ",
    "accuracy : ", str(accuracy_score(y_b, pred))+" \n",
    "precision : " + str(precision_score(y_b, pred, average='weighted'))+" \n",
    "recall : "+ str(recall_score(y_b, pred, average='weighted')) +" \n",
    "f1 : " + str(f1_score(y_b, pred, average='weighted'))+" \n"]
    results_cross.writelines(txt)
    results_cross.close()


def simple_recurrent(datasets):
    splits=10
    batch_size=128

    for dataset in datasets:
        print("Evaluating : ", dataset  )

        X,y = load_csv(dataset)
        X = get_binary_words(X)

        results = train_and_evaluate(X,y,batch_size,splits,simple=True)

        print("mean metrics cv=10")
        print("accuracy : mean={}, std={}".format(np.mean(results[0]),np.std(results[0])))
        print("precision : mean={}, std={}".format(np.mean(results[1]),np.std(results[1])))
        print("recall : mean={}, std={}".format(np.mean(results[2]),np.std(results[2])))
        print("f1 : mean={}, std={}".format(np.mean(results[3]),np.std(results[3])))
        print("\n")

    

if __name__ == '__main__':
    datasets=["H3-clean.csv","H4-clean.csv","H3K4me1-clean.csv",
        "H3K4me2-clean.csv","H3K4me3-clean.csv","H3K9ac-clean.csv",
        "H3K14ac-clean.csv","H3K36me3-clean.csv","H3K79me3-clean.csv",
        "H4ac-clean.csv","splice.csv"]

    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_dataset', help='input file', required=False)
    parser.add_argument('-b', '--batch_size', help='input file', required=False)
    parser.add_argument('-c', '--cross_dataset', help='cross dataset evaluation', required=False)

    ARGS = parser.parse_args()

    #X,y = load_csv(ARGS.input_dataset)
    splits=10

    if ARGS.batch_size:
        batch_size=ARGS.batch_size
    else:
        batch_size=128

    #X = get_binary_words(X)

    if ARGS.cross_dataset:
        print("cross dataset experiment")
        

        model = cross_dataset_train(X,y,batch_size)

        for dataset in datasets:
            if dataset == ARGS.input_dataset:
                continue
            print("evaluating : ",dataset)
            X_b,y_b=load_csv(dataset)
            X_b = get_binary_words(X_b,flatten_words=True)

            cross_dataset_evaluation(model,X_b,y_b)

        exit()

    for dataset in datasets:
        print("Evaluating , ",dataset)
        X,y = load_csv(dataset)

        X = get_binary_words(X)

        results = train_and_evaluate(X,y,batch_size,splits)

        print("mean metrics cv=10")
        print("accuracy : mean={}, std={}".format(np.mean(results[0]),np.std(results[0])))
        print("precision : mean={}, std={}".format(np.mean(results[1]),np.std(results[1])))
        print("recall : mean={}, std={}".format(np.mean(results[2]),np.std(results[2])))
        print("f1 : mean={}, std={}".format(np.mean(results[3]),np.std(results[3])))
        print("\n")





