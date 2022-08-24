#in order to import datasets, you need git clone https://github.com/srndic/mimicus.git
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch import optim,nn
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data.dataset import Dataset
import torch.nn as nn
import torch.optim as optim
from torch.utils.data.dataset import Dataset
import csv

import numpy
def csv2numpy(csv_in):
    '''
    Parses a CSV input file and returns a tuple (X, y) with
    training vectors (numpy.array) and labels (numpy.array), respectfully.

    csv_in - name of a CSV file with training data points;
                the first column in the file is supposed to be named
                'class' and should contain the class label for the data
                points; the second column of this file will be ignored
                (put data point ID here).
    '''
    # Parse CSV file
    csv_rows = list(csv.reader(open(csv_in, 'r')))
    classes = {'FALSE': 0, 'TRUE': 1}
    rownum = 0
    # Count exact number of data points
    TOTAL_ROWS = 0
    for row in csv_rows:
        if row[0] in classes:
            # Count line if it begins with a class label (boolean)
            TOTAL_ROWS += 1
    # X = vector of data points, y = label vector
    X = numpy.array(numpy.zeros((TOTAL_ROWS, 135)), dtype=numpy.float64, order='C')
    y = numpy.array(numpy.zeros(TOTAL_ROWS), dtype=numpy.float64, order='C')
    file_names = []
    for row in csv_rows:
        # Skip line if it doesn't begin with a class label (boolean)
        if row[0] not in classes:
            continue
        # Read class label from first row
        y[rownum] = classes[row[0]]
        featnum = 0
        file_names.append(row[1])
        for featval in row[2:]:
            if featval in classes:
                # Convert booleans to integers
                featval = classes[featval]
            X[rownum, featnum] = float(featval)
            featnum += 1
        rownum += 1
    return X, y, file_names
class DatasetFromCSV(Dataset):
    def __init__(self, csv_path):
        self.data, self.label, _ = csv2numpy(csv_path)

    def __getitem__(self, index):
        return (self.data[index], self.label[index])

    def __len__(self):
        return len(self.data)