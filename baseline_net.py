
import csv
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical
from sklearn.model_selection import StratifiedKFold
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
        #print("accuracy : ", accuracy_score(y_test, pred))

        accs.append(accuracy_score(y_test, pred))
        pres.append(precision_score(y_test, pred, average='weighted'))
        recalls.append(recall_score(y_test, pred, average='weighted'))
        f1s.append(f1_score(y_test, pred, average='weighted'))
        fold+=1


        del model

    results = [accs,pres,recalls,f1s]

    return results

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

        model, epochs = create_model(y_train.shape[1],X_train.shape[1:])
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


if __name__ == '__main__':
    datasets=["H3-clean.csv","H4-clean.csv","H3K4me1-clean.csv",
        "H3K4me2-clean.csv","H3K4me3-clean.csv","H3K9ac-clean.csv",
        "H3K14ac-clean.csv","H3K36me3-clean.csv","H3K79me3-clean.csv",
        "H4ac-clean.csv","splice.csv"]


    splits=10
    batch_size = 128

    for dataset in datasets:
        print("Evaluating , ",dataset)
        X, y = [], []
        classes = -1
        available = []
        input_file = "datasets/original/" + dataset

        with open(input_file) as raw_data:
            data = csv.reader(raw_data, delimiter=',')
            for idx, val in enumerate(data):
                if idx == 0:
                    continue
                sequence = val[2]
                target = val[0]
                preprocessed = sequence.replace(" ", "")

                X.append(preprocessed)
                if not target in available:
                    classes += 1
                    y.append(classes)
                    available.append(target)
                else:
                    y.append(classes)

        X = get_binary_words(X)
        results = train_and_evaluate(X,y,batch_size,splits)

        print("mean metrics cv=10")
        print("accuracy : mean={}, std={}".format(np.mean(results[0]),np.std(results[0])))
        print("precision : mean={}, std={}".format(np.mean(results[1]),np.std(results[1])))
        print("recall : mean={}, std={}".format(np.mean(results[2]),np.std(results[2])))
        print("f1 : mean={}, std={}".format(np.mean(results[3]),np.std(results[3])))
        print("\n")
