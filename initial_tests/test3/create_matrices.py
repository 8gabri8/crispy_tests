import os
import numpy as np
import scipy
from matplotlib import pyplot as plt
import gzip
import pandas as pd
from nigsp import io, viz
from nigsp.operations.timeseries import resize_ts


"""
    creates 2 structral matrices, one with no motort neuron connection, and one with only that

    create also the corrisponding I
"""
regions = np.array(range(130,150))

#create 2 partials strucrral matrices
s = io.load_mat('../SC_avg56.mat')
s_all = np.copy(s) ; s_all[:, regions] = 0 ; s_all[regions, :] = 0
s_few = np.zeros(s.shape) ; print(s_few.shape)
for i in range(s.shape[0]):
    for j in range(s.shape[1]):
        if i in regions or j in regions:
            s_few[i, j] = s[i, j]
io.export_mtx(s_all,'s_all.mat')
io.export_mtx(s_few,'s_few.mat')

#create 2 partials Identity matrices
I_all = np.zeros(s.shape)
I_few = np.zeros(s.shape) #make I the same dimension of structrul matrix
for i in range(int(s.shape[0])):
    if i in regions: #i am in the few region
        I_few[i,i] = 1
    else:
        I_all[i,i] = 1
io.export_mtx(I_all,'I_all.mat')
io.export_mtx(I_few,'I_few.mat')

fig, axes = plt.subplots(2, 2, figsize=(10, 10))
pos_0 = axes[0,0].imshow(np.log(s_all), interpolation='nearest', vmin=-1, vmax=1); axes[0,0].set_title("log(s_all)")
pos_1 =axes[0,1].imshow(np.log(s_few), interpolation='nearest', vmin=-1, vmax=1); axes[0,1].set_title("log(s_few)")
pos_2 = axes[1,0].imshow(I_all, interpolation='nearest', vmin=0, vmax=1); axes[1,0].set_title("I_all")
pos_3 = axes[1,1].imshow(I_few, interpolation='nearest', vmin=0, vmax=1); axes[1,1].set_title("I_few")

#fig.colorbar(pos_0, ax=axes[0]); fig.colorbar(pos_1, ax=axes[1]); #fig.colorbar(pos_2, ax=axes[2])
plt.tight_layout()
fig.savefig("matrices.png")
plt.show() 
