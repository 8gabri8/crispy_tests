import os
import numpy as np
import scipy
from matplotlib import pyplot as plt
import gzip
import pandas as pd
from nigsp import io, viz
from nigsp.operations.timeseries import resize_ts


"""
    similiar of test 2, but this time verify what happnes for realisti situation (ex. only remove the motor region)
    we will run
        --> with the all brain, but some nodes left
        --> with only the removed nodes

    -not0 -s ... I
"""

#load the mactrices
s_all = io.load_mat('s_all.mat')
s_few = io.load_mat('s_few.mat')
I_all = io.load_mat('I_all.mat')
I_few = io.load_mat('I_few.mat')


os.system("python3 ../crispy_gls_scalar.py -not0 -s s_all.mat I_all.mat -f ../RS_1subj.mat -sub 1 -od all")
os.system("python3 ../crispy_gls_scalar.py -not0 -s s_few.mat I_few.mat -f ../RS_1subj.mat -sub 1 -od few")

#load E
E_all = io.load_txt('all/files/sub-1_ts-innov.tsv.gz')
E_few = io.load_txt('few/files/sub-1_ts-innov.tsv.gz')

#create Ydiff
f = io.load_mat('../RS_1subj.mat') #print(f.shape) #-->FUNCTIONAL data
f = resize_ts(f, resize="norm")
f = f - f.mean(axis=0)[np.newaxis, ...]
Ydiff = np.diff(f) ; print(Ydiff.shape)

#plot E, Ydiff and E-Ydiff
fig, axes = plt.subplots(3, 1, figsize=(10, 10))
pos_0 = axes[0].imshow(np.abs(E_all-Ydiff), interpolation='nearest', vmin=0, vmax=0.5); axes[0].set_title("abs(E_all-Ydiff)")
pos_1 = axes[1].imshow(np.abs(E_few-Ydiff), interpolation='nearest', vmin=0, vmax=0.5); axes[1].set_title("abs(E_few-Ydiff)")
pos_2 = axes[2].imshow(np.abs(E_all-Ydiff) - np.abs(E_few-Ydiff), interpolation='nearest', vmin=0, vmax=0.5); axes[2].set_title("np.abs(E_all-Ydiff) - np.abs(E_few-Ydiff)")
fig.colorbar(pos_0, ax=axes[0]); fig.colorbar(pos_1, ax=axes[1]); fig.colorbar(pos_2, ax=axes[2])
plt.tight_layout()
fig.savefig("all_Vs_few.png")
plt.show() 

#viz.plot_connectivity(s, "nigsp.png")

