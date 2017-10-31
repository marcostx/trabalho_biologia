import pandas as pd
import sys
import numpy as np
import matplotlib.pyplot as plt

import os


def load_csv(input_file='dataset.csv'):
    X,y=[],[]
    classes=0
    avaible = []

    raw_data = pd.read_csv(input_file)
    for idx, val in enumerate(raw_data.values):
    	X.append(raw_data.sequence[idx])
    	if not raw_data.target[idx] in avaible:
    		avaible.append(raw_data.target[idx])
    		classes+=1
    		y.append(classes)
    	else:
    		y.append(classes)



#if __name__ == '__main__':
#	dataset = load_csv(sys.argv[1])

	#print(dataset.sequence)