import os
import numpy as np
import scipy
from matplotlib import pyplot as plt
import copy
from nigsp import io

"""
    create a 6x6 structrual matrix (subbart of the big one) with a time seire s(fmri) random of 200 timepoints
"""

#create fake data
s = io.load_mat('../SC_avg56.mat')
my_s = np.copy(s[:6,:6])
io.export_mtx(my_s,'S_small.mat')

f = np.random.randn(6,200)
io.export_mtx(f,'random_fMRI.mat')

#calculate tau for random data
os.system("python3 ../crispy_gls_scalar.py -s S_small.mat -f random_fMRI.mat -sub 1 -od random")

#load E
E = io.load_txt('random/files/sub-1_ts-innov.tsv.gz')

fig, axes = plt.subplots(2, 1, figsize=(10, 10))
pos_0 = axes[0].imshow(f, interpolation='nearest', vmin=-1, vmax=1); axes[0].set_title("fMRI time serie")
pos_1 =axes[1].imshow(E, interpolation='nearest', vmin=-1, vmax=1); axes[1].set_title("E")
fig.colorbar(pos_0, ax=axes[0]); fig.colorbar(pos_1, ax=axes[1]); #fig.colorbar(pos_2, ax=axes[2])
plt.tight_layout()
fig.savefig("random_data.png")
plt.show() 
