import collections, sys
from Bio import Seq, SeqIO, SeqRecord
import itertools
from parser import load_csv
import numpy as np
import argparse
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.constraints import maxnorm
from keras.optimizers import SGD
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
import itertools
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score
from keras.utils import np_utils
from keras.layers.recurrent import LSTM
from keras.layers import SimpleRNN
from keras import initializers
from utils import *


def create_baseline_model(num_classes,inp_shape):
	
	model = Sequential()
	model.add(Conv2D(32, kernel_size=(1, 1),activation='relu', input_shape=(inp_shape[1], inp_shape[0], inp_shape[2])))
	model.add(MaxPooling2D(pool_size=(1, 1)))
	model.add(Conv2D(32, kernel_size=(2, 2),activation='relu'))
	model.add(MaxPooling2D(pool_size=(1, 1)))
	model.add(Flatten())
	model.add(Dense(100, activation='relu'))
	model.add(Dropout(0.5))
	model.add(Dense(num_classes, activation='softmax'))
	epochs = 120
	lrate = 0.001
	decay = lrate/epochs
	sgd = Adam(lr=lrate, epsilon=1e-08, decay=decay)
	model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

	return model, epochs

def create_reccurent_model(num_classes,inp_shape):
	epochs = 120
	batch_size=32
	lrate=0.001
	decay = lrate/epochs

	model = Sequential()
	model.add(LSTM(32, return_sequences=False,
               input_shape=inp_shape[1:]))
	model.add(Dense(num_classes, activation='softmax'))
	
	sgd = Adam(lr=lrate, epsilon=1e-08, decay=decay)
	model.compile(loss='categorical_crossentropy',
	              optimizer=sgd,
	              metrics=['accuracy'])
	

	return model, epochs

if __name__ == '__main__':

	#assert len(sys.argv[1]) > 1
	parser = argparse.ArgumentParser()
	parser.add_argument('-i', '--input_dataset', help='input file', required=True)
	parser.add_argument('-m', '--method', help='method', required=True)
	ARGS = parser.parse_args()


	X,y = load_csv(ARGS.input_dataset)
	

	X_binarized = preprocess(X)
	inp_shape = X_binarized.shape
	

	X_train, X_test, y_train, y_test = train_test_split(X_binarized, y, test_size=0.33, random_state=42)

	y_train = np_utils.to_categorical(y_train)
	y_test_ = np_utils.to_categorical(y_test)

	num_classes = y_train.shape[1]

	if ARGS.method == 'baseline':
		model,epochs = create_baseline_model(num_classes,inp_shape)
	elif ARGS.method == 'recurrent':
		model,epochs = create_reccurent_model(num_classes,inp_shape)

	seed = 7
	np.random.seed(seed)
	
	model.fit(X_train, y_train, validation_data=(X_test, y_test_), nb_epoch=epochs, batch_size=50)
	# Final evaluation of the model
	pred = model.predict(X_test, verbose=0)
	pred = [np.argmax(item) for item in pred]

	print("accuracy : ", accuracy_score(y_test, pred ) )
	print("precision : ", precision_score(y_test, pred, average='weighted' ) )
	print("recall : ", recall_score(y_test, pred,average='weighted' ) )
	print("f1 : ", f1_score(y_test, pred,average='weighted' ) )
	print("\n")



	
	
