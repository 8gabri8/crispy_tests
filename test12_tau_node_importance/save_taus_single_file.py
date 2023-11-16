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
    take the 2 taus calculated in lesion-paz (CALCULATED TOGETEHR) and put them ALL of them in a single file,
    one for connecotme and one for lesione
"""

s = io.load_mat('raw/SC_avg56.mat')

taus_connectome = []
taus_lesion = []


for i in range(s.shape[0]): #360 nodes
    taus = np.array(io.load_txt(f"data/test12/paz-lesion/paz-lesion_{i}/files/sub-1_tau_scalar.tsv"))
    taus_lesion.append(taus[0])
    taus_connectome.append(taus[1])

print(len(taus_connectome), len(taus_lesion))

io.export_mtx(taus_connectome, f'raw/taus_connectome.mat')
io.export_mtx(taus_lesion, f'raw/taus_lesion.mat')



