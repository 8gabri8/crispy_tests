# import os
import numpy as np
# import scipy
from matplotlib import pyplot as plt
# import gzip
# import pandas as pd
# import copy
from nigsp import io
from crispy import crispy_gls_scalar, crispy_var_model
from nigsp.operations.metrics import functional_connectivity
from nigsp.viz import plot_connectivity
from nigsp.operations import laplacian
from nilearn.plotting import plot_matrix



"""
    Try the crispy_var_model script
    NOT put -not0 --> so I want tau0

    try both cripsy_scalar with normalized and not normlaized structral matrices

    use in cripsy_scalar as structral matrix the correlation of functional ts of single voxle (indeed that also is a type of functional connectivity)

"""
calculations = False
#matrix used for connecitvut matrix
c = "data/test13/functional_connectivity"

#calculate functional connectivity
ts = io.load_mat("raw/RS_1subj.mat") #load fMRI ts
fs = functional_connectivity(ts) #calculate fun_conn (corr of easch 2 ts of a single voxels)
fs = np.abs(fs) #negative connecitviyt doesnt exists
plot_connectivity(fs, f"{c}.png")
io.export_mtx(fs,f"{c}.mat" )

if calculations:
    crispy_var_model.VAR_estimation(tsfile = "raw/RS_1subj.mat",
                    sub = "1",
                    odr = "data/test13/var_model")

    crispy_gls_scalar.multitau_gls_estimation(tsfile = "raw/RS_1subj.mat",
                    structural_files = f"{c}.mat",
                    sub = "1",
                    odr = "data/test13/norm_diffusion_model")

    crispy_gls_scalar.multitau_gls_estimation(tsfile = "raw/RS_1subj.mat",
                    structural_files = f"{c}.mat",
                    structural_files_nonorm = f"{c}.mat",
                    sub = "1",
                    odr = "data/test13/not_norm_diffusion_model")

#check if A == (I - tao0 I - tau1 L1)
    #USE L1 NOT NORMALISED
    #USE TAUS FROM YES NORMALISED
A = io.load_txt("data/test13/var_model/files/sub-1_Amat.txt")
conn_mat = io.load_mat(f"{c}.mat")
taus = io.load_txt("data/test13/norm_diffusion_model/files/sub-1_tau_scalar.tsv")
t0 = taus[0]
t1 = taus[1]
n = A.shape[0]
I = np.identity(n)
L1, degree = laplacian.compute_laplacian(conn_mat, selfloops="degree")
L1_norm = laplacian.normalisation(conn_mat, degree, norm="rwo")

M = I - t0 * I - t1 * L1

fig, ax = plt.subplots(1,3, figsize = (15,10))
ax[0].imshow(A, interpolation='nearest', aspect="auto"); ax[0].set_title(f"A"); ax[0].tick_params(left = False, right = False , labelleft = False , labelbottom = False, bottom = False) 
ax[1].imshow(M, interpolation='nearest', aspect="auto"); ax[1].set_title(f"M = I - t0 * I - t1 * L1"); ax[1].tick_params(left = False, right = False , labelleft = False , labelbottom = False, bottom = False) 
ax[2].imshow(M-A, interpolation='nearest', aspect="auto"); ax[2].set_title(f"M - A"); ax[2].tick_params(left = False, right = False , labelleft = False , labelbottom = False, bottom = False) 

fig.savefig("data/test13/A_M.png")
plt.show()

######
#plot the diffenrt errors
E_var = io.load_txt("data/test13/var_model/files/sub-1_ts-innov.txt")
E_diff = io.load_txt("data/test13/norm_diffusion_model/files/sub-1_ts-innov.tsv.gz")
E_diff_not_norm = io.load_txt("data/test13/not_norm_diffusion_model/files/sub-1_ts-innov.tsv.gz")
print(type(E_diff_not_norm[0,0]))


fig, ax = plt.subplots(3,2, figsize = (15,15))
ax[0,0].imshow(E_var, interpolation='nearest', aspect="auto"); ax[0,0].set_title(f"E_var"); ax[0,0].tick_params(left = False, right = False , labelleft = False , labelbottom = False, bottom = False) 
ax[1,0].imshow(E_diff, interpolation='nearest', aspect="auto"); ax[1,0].set_title(f"E_diff"); ax[1,0].tick_params(left = False, right = False , labelleft = False , labelbottom = False, bottom = False) 
pos1 = ax[2,0].imshow(np.abs(E_var-E_diff), interpolation='nearest', aspect="auto"); ax[2,0].set_title(f"abs(E_var-E_diff)"); ax[2,0].tick_params(left = False, right = False , labelleft = False , labelbottom = False, bottom = False) 

ax[0,1].imshow(E_var, interpolation='nearest', aspect="auto"); ax[0,1].set_title(f"E_var"); ax[0,1].tick_params(left = False, right = False , labelleft = False , labelbottom = False, bottom = False) 
ax[1,1].imshow(E_diff_not_norm, interpolation='nearest', aspect="auto"); ax[1,1].set_title(f"E_diff_not_norm"); ax[1,1].tick_params(left = False, right = False , labelleft = False , labelbottom = False, bottom = False) 
pos2 = ax[2,1].imshow(np.abs(E_var-E_diff_not_norm), interpolation='nearest', aspect="auto"); ax[2,1].set_title(f"abs(E_var-E_diff_not_norm)"); ax[2,1].tick_params(left = False, right = False , labelleft = False , labelbottom = False, bottom = False) 

plt.colorbar(pos1, ax=ax[2,0])
plt.colorbar(pos2, ax=ax[2,1])



fig.savefig("data/test13/errors.png")
