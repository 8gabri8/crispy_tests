
import numpy as np
from matplotlib import pyplot as plt
from nigsp import io
from crispy import crispy_gls_scalar, crispy_var_model
from nigsp.operations.metrics import functional_connectivity
import os
import scipy.io
import csv
import nigsp
# from nigsp.viz import plot_connectivity
from nigsp.operations import laplacian
# from nilearn.plotting import plot_matrix
from numpy.linalg import norm
from nigsp.operations.timeseries import resize_ts
import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from nigsp import io, viz
from nigsp.operations import laplacian
from nigsp.operations.timeseries import resize_ts

"""
    tau1 and tau1 claculated in test 12 together (no tau0)
    verify if the error is smaller if i use just a sigle tau insted of 2 --> confrotn the e Errors
        E 2 taus already calculated
        E 1 taius to calculet --> E = Ydiff + tauLY
"""

#structural matrix
s = io.load_mat('raw/SC_avg56.mat')

data_path = "data/test18"
if not os.path.exists(data_path):
    os.makedirs(data_path)

def order_taus(index):
    taus_1_paz_lesion = [] #is lesion
    taus_2_paz_lesion = [] #is connectome

    for node in index: #NB index is an array with values between 0 and 359 (one for each node)
        taus = np.array(io.load_txt(f"data/test12/paz-lesion/paz-lesion_{node}/files/sub-1_tau_scalar.tsv"))
        taus_1_paz_lesion.append(taus[0])
        taus_2_paz_lesion.append(taus[1])

    return taus_1_paz_lesion, taus_2_paz_lesion

#oerder of nodes
index = range(360)

#exctart taus ech node
taus_1_paz_lesion, taus_2_paz_lesion = order_taus(index)

# laplacin of structral matrix
s = io.load_mat('raw/SC_avg56.mat')
L, degree = laplacian.compute_laplacian(s, selfloops="degree") #THEY ARE NOT NORMALISED
L = laplacian.normalisation(L, degree, norm="rwo")

#import time seri
ts = io.load_mat("raw/RS_1subj.mat")
# Column-center ts (sort of GSR)
ts = resize_ts(ts, resize="norm")
ts = ts - ts.mean(axis=0)[np.newaxis, ...]

#ectrct the error using 2 taus
calculation = 0

if calculation:
    E = np.zeros(s.shape[0])
    for i in range(s.shape[0]):
        print(f"calculatin error {i}")
        E[i] = norm(io.load_txt(f"data/test12/paz-lesion/paz-lesion_{i}/files/sub-1_ts-innov.tsv.gz"))
    io.export_mtx(E, "data/test12/norms_errors_paz-lesion.mat")

norm_E_double = io.load_mat("data/test12/norms_errors_paz-lesion.mat")

#####################
### tau = tau1 + tau2
####################
norm_E_single = np.zeros(s.shape[0])

ts_diff = np.diff(ts)
ts_cut = ts[:, 1:]
print(ts_diff.shape, ts_cut.shape)

for i in range(s.shape[0]):
    t = taus_1_paz_lesion[i] + taus_2_paz_lesion[i]
    norm_E_single[i] = norm(ts_diff + t * L @ ts_cut)

fig, a = plt.subplots(1,1, dpi=300)
a.plot(norm_E_double,label="nom Error two taus")
a.plot(norm_E_single,label="nom Error one taus SUM")

plt.legend()
plt.tight_layout()
plt.savefig("data/test18/E_sum.png")

#####################
### tau = meean(tau1 + tau2)
####################
norm_E_single = np.zeros(s.shape[0])

ts_diff = np.diff(ts)
ts_cut = ts[:, 1:]
print(ts_diff.shape, ts_cut.shape)

for i in range(s.shape[0]):
    t = np.mean([taus_1_paz_lesion[i], taus_2_paz_lesion[i]])
    norm_E_single[i] = norm(ts_diff + t * L @ ts_cut)

fig, a = plt.subplots(1,1, dpi=300)
a.plot(norm_E_double,label="nom Error two taus")
a.plot(norm_E_single,label="nom Error one taus MEAN")

plt.legend()
plt.tight_layout()
plt.savefig("data/test18/E_mean.png")