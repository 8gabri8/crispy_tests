import numpy as np
from matplotlib import pyplot as plt
from nigsp import io
from crispy import crispy_gls_scalar, crispy_var_model
from nigsp.operations.metrics import functional_connectivity
import os
import scipy.io
import csv
import nigsp

"""
    take the file that associate each node to a FN
    change the lables of the 3 node that are in the FN8
    300 120 -- in FN (5)
    299(left pre subiculum) 119 -- in Fn (1)

    now labels are 0 to 6 (7FN)
"""

# Load .mat file
mat = scipy.io.loadmat('raw/Glasser_ch2_yeo_RS7.mat')
print(mat.keys())
data = mat['yeoROIs'] #data --> array of arraus of single elements, the single lemts are label of in which FN is the specific node
print("before", np.unique(data))
print("before", np.where(data == 8))

data[299] = 5
data[119] = 5
data[298] = 1 #(left pre subiculum)
data[118] = 1 

print("after", np.unique(data))
print("after", np.where(data == 8))

data = data - 1

io.export_mtx(data, f'raw/FN_labels.mat')

####################################################àà
# create a list that contains the indices of each
n = len(np.unique(data)) #number of FN
indices = []
for i in range(n):
    idx, value = np.where(data == i)
    indices.append(idx)

print(indices)
for i in range(n):
    print(len(indices[i]))


