import os
import numpy as np
import scipy
from matplotlib import pyplot as plt
import copy
from nigsp import io, viz

"""
    create 360 structural matrices (one for each node)
    the matrix of the node i: is 0 everywhere, except on the row and column i. NB here the weights are halved, or we count twice
    in this way the sum of alla this matrices is the origina stricutral/adjacnency matrix 

    the result found in this way (decompostion of strucutral matrx) is the same of when we pass the entire structural matrix??
"""
#because is long to run for all tau, just tun once with the tau calculatio
tau_calculation = False

s = io.load_mat('../SC_avg56.mat')
k = []

for i in range(s.shape[0]):
    mat = np.zeros([int(s.shape[0]),int(s.shape[0])])
    mat[:,i] = s[:,i]/2 #i-th colums
    mat[i,:] = s[i,:]/2
    io.export_mtx(mat,f'matrices/node_{i}.mat')
    if i == 300:
        k = mat


if tau_calculation:
    # execute with a single matrix per node
    os.system("python3 ../crispy_gls_scalar.py -s matrices/* -f ../RS_1subj.mat -sub 1 -od single_node")

    # execute with the total adjacency matrix
    os.system("python3 ../crispy_gls_scalar.py -s ../SC_avg56.mat -f ../RS_1subj.mat -sub 1 -od total_adjacency")

#if the 2 methos are the same, E1 == E2
E_single = io.load_txt("single_node/files/sub-1_ts-innov.tsv.gz")
E_all = io.load_txt("total_adjacency/files/sub-1_ts-innov.tsv.gz")


#plot E, Ydiff and E-Ydiff
fig, axes = plt.subplots(3, 1, figsize=(10, 10))
pos_0 = axes[0].imshow(E_single, interpolation='nearest', vmin=-1, vmax=1); axes[0].set_title("E_single")
pos_1 =axes[1].imshow(E_all, interpolation='nearest', vmin=-1, vmax=1); axes[1].set_title("E_all")
pos_2 = axes[2].imshow(np.abs(E_single-E_all), interpolation='nearest', vmin=0, vmax=1); axes[2].set_title("ABS(E_single - E_all) ")
fig.colorbar(pos_0, ax=axes[0]); fig.colorbar(pos_1, ax=axes[1]); fig.colorbar(pos_2, ax=axes[2])
plt.tight_layout()
fig.savefig("single_Vs_all.png")
plt.show() 

fig, axes = plt.subplots(1, 1, figsize=(10, 10))
axes.imshow(np.log(k), interpolation='nearest', vmin=0, vmax=1, aspect="auto"); axes.set_title(f""); axes.tick_params(left = False, right = False , labelleft = False , labelbottom = False, bottom = False)
plt.tight_layout()
fig.savefig("single.png")
plt.show() 