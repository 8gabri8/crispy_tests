import os
import numpy as np
import scipy
from matplotlib import pyplot as plt
import gzip
import pandas as pd
import copy
from nigsp import io


"""
    verify if -not0 -s S I == -s S
"""

print("calculating with tau0")
os.system("python3 ../crispy_gls_scalar.py -s ../SC_avg56.mat -f ../RS_1subj.mat -sub 1 -od yes_tau_0")

#create identity matrix
s = io.load_mat('../SC_avg56.mat') #NB nigsp io --> load directlythe matrix as nd.array
I = np.identity(s.shape[0]) #make I the same dimension of structrul matrix
# temp = copy.deepcopy(s_dict)
# del temp["SC_avg56"]
# temp["I"] = I
# scipy.io.savemat('I.mat', temp) #matrix must be saved as dictionary
io.export_mtx(I,'I.mat')

print("calculating without tau0")
os.system("python3 ../crispy_gls_scalar.py -not0 -s ../SC_avg56.mat I.mat -f ../RS_1subj.mat -sub 1 -od no_tau_0")

#some checks visual
#plt.imshow(I, interpolation='nearest'); plt.show()
#print(s.shape, I.shape)

#verify if the inoovation are the same
#TSV are like CSV, they save the matrix (NxT)
no_tau0 = io.load_txt('no_tau_0/files/sub-1_ts-innov.tsv.gz')
yes_tau0 = io.load_txt('yes_tau_0/files/sub-1_ts-innov.tsv.gz')

fig, axes = plt.subplots(3, 1, figsize=(10, 10))
pos_0 = axes[0].imshow(no_tau0, interpolation='nearest', vmin=-1, vmax=1); axes[0].set_title("E no_tau0")
pos_1 =axes[1].imshow(yes_tau0, interpolation='nearest', vmin=-1, vmax=1); axes[1].set_title("E yes_tau0")
pos_2 = axes[2].imshow(np.abs(no_tau0 - yes_tau0), interpolation='nearest', vmin=0, vmax=1); axes[2].set_title("ABS(E no_tau0 - E yes_tau0) ")
fig.colorbar(pos_0, ax=axes[0]); fig.colorbar(pos_1, ax=axes[1]); fig.colorbar(pos_2, ax=axes[2])
print(np.mean(np.abs( no_tau0 - yes_tau0 )))
plt.tight_layout()
plt.show()
print(no_tau0.shape) #--> (359, 1189)
print(yes_tau0.shape)
print(np.max(np.abs(no_tau0 - yes_tau0)))

fig.savefig("yes_no_tau0.png")