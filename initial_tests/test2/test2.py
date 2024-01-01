import os
import numpy as np
import scipy
from matplotlib import pyplot as plt
import gzip
import pandas as pd
from nigsp import io, viz
from nigsp.operations.timeseries import resize_ts


"""
    verify if -not0 S2 Ipar is the righ way to create the case where an entire part of ther brain doesn't diffuse
"""

#create PARTIAL identity matrix
s = io.load_mat('../SC_2.mat')
I = np.identity(s.shape[0]) #make I the same dimension of structrul matrix
for i in range(int(I.shape[0]/2)):
    I[i,i] = 0
io.export_mtx(I,'I.mat')

print("calculating...")
os.system("python3 ../crispy_gls_scalar.py -not0 -s ../SC_2.mat I.mat -f ../RS_1subj.mat -sub 1 -od only_dx_hemisphere")

#some checks visual
#print(sum(sum(I))) #--->180 ones
fig, axes = plt.subplots(1,2, figsize=(10, 10))
axes[0].imshow(I, interpolation='nearest', vmin=0, vmax=1); 
axes[1].imshow(s, interpolation='nearest', vmin=0, vmax=1); 
fig.savefig("matrices.png")
plt.show()

#IF the signal has not diffused in the sx hemisphere, we should see E == Ydiff in that part - so their difference ==0
f = io.load_mat('../RS_1subj.mat') #print(f.shape) #-->FUNCTIONAL data

# Column-center ts (sort of GSR)
f = resize_ts(f, resize="norm")
f = f - f.mean(axis=0)[np.newaxis, ...]

Ydiff = np.diff(f) ; print(Ydiff.shape)

#load E
E = io.load_txt('only_dx_hemisphere/files/sub-1_ts-innov.tsv.gz')

#plot E, Ydiff and E-Ydiff
fig, axes = plt.subplots(3, 1, figsize=(10, 10))
pos_0 = axes[0].imshow(Ydiff, interpolation='nearest', vmin=-1, vmax=1); axes[0].set_title("Ydiff")
pos_1 =axes[1].imshow(E, interpolation='nearest', vmin=-1, vmax=1); axes[1].set_title("E")
pos_2 = axes[2].imshow(np.abs(Ydiff-E), interpolation='nearest', vmin=0, vmax=0.2); axes[2].set_title("ABS(E - Ydiff) ")
fig.colorbar(pos_0, ax=axes[0]); fig.colorbar(pos_1, ax=axes[1]); fig.colorbar(pos_2, ax=axes[2])
plt.tight_layout()
fig.savefig("half_brain_diffusion.png")
plt.show() 

#viz.plot_connectivity(s, "nigsp.png")

