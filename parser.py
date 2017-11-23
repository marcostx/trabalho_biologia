import sys
import numpy as np
import matplotlib.pyplot as plt

import os


def load_csv(input_file='dataset.csv'):
    X,y=[],[]
    classes=-1
    available = []

    #raw_data = pd.read_csv(input_file)
    with open(input_file) as raw_data:
        for idx, val in enumerate(raw_data.readlines()):
            sequence = val.split(",")[2]
            target   = val.split(",")[0]

            # removing \t\t
            preprocessed = sequence.replace("\t","")
            preprocessed = preprocessed.replace(" ","")

            X.append(preprocessed)
            if not target in available:
                classes += 1
                y.append(classes)
                available.append(target)
            else:
                y.append(classes)

    
    return X,np.array(y)