import pandas as pd
import sys
import numpy as np
import matplotlib.pyplot as plt

import os


def load_csv(input_file='dataset.csv'):
    X,y=[],[]
    classes=-1
    available = []

    raw_data = pd.read_csv(input_file)
    for idx, val in enumerate(raw_data.values):

        # removing \t\t
        preprocessed = raw_data.sequence[idx].replace("\t","")
        preprocessed = preprocessed.replace(" ","")
        X.append(preprocessed)
        if not raw_data.target[idx] in available:
            classes += 1
            y.append(classes)
            available.append(raw_data.target[idx])
            

        else:
            y.append(classes)

    return X,np.array(y)
#if __name__ == '__main__':
#	dataset = load_csv(sys.argv[1])

    #print(dataset.sequence)