import sys
import numpy as np
import matplotlib.pyplot as plt
import csv
import os


def load_csv(input_file='dataset.csv'):
    X,y=[],[]
    classes=-1
    available = []
    #input_file = "./datasets/original/"+input_file

    #raw_data = pd.read_csv(input_file)
    with open(input_file) as raw_data:
        data = csv.reader(raw_data, delimiter=',')

        for idx, val in enumerate(data):
            if idx==0:
                continue
            sequence = val[2]
            target   = val[0]

            # removing \t\t
            
            preprocessed = sequence.replace(" ","")
            #preprocessed = preprocessed[1:-2]
            #print(preprocessed)

            X.append(preprocessed)
            if not target in available:
                classes += 1
                y.append(classes)
                available.append(target)
            else:
                y.append(classes)

    
    return X,np.array(y)
