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
    NOT put -not0 --> so I want tau0 (yes self loop)

    try both cripsy_scalar with normalized and not normlaized structral matrices

    use in cripsy_scalar as structral matrix the correlation of functional ts of single voxle (indeed that also is a type of functional connectivity)

"""
calculations = False
#matrix used for connecitvut matrix
c = "raw/SC_avg56.mat"

#calculate functional connectivity
ts = io.load_mat("raw/RS_1subj.mat") #load fMRI ts
fs = functional_connectivity(ts) #calculate fun_conn (corr of easch 2 ts of a single voxels)
fs = np.abs(fs) #negative connecitviyt doesnt exists
#plot_connectivity(fs, f"{c}.png")
#io.export_mtx(fs,f"{c}.mat" )

if calculations:
    crispy_var_model.VAR_estimation(tsfile = "raw/RS_1subj.mat", #inside usese python module
                    sub = "1",
                    odr = "data/test13/var_model")

    crispy_gls_scalar.multitau_gls_estimation(tsfile = "raw/RS_1subj.mat",
                    structural_files = f"{c}",
                    sub = "1",
                    odr = "data/test13/norm_diffusion_model")

    crispy_gls_scalar.multitau_gls_estimation(tsfile = "raw/RS_1subj.mat",
                    structural_files = f"{c}",
                    structural_files_nonorm = f"{c}.mat",
                    sub = "1",
                    odr = "data/test13/not_norm_diffusion_model")

###### LOAD RESULTS
conn_mat = io.load_mat(f"{c}")

# results from VAR
A = io.load_txt("data/test13/var_model/files/sub-1_Amat.txt")
E_var = io.load_txt("data/test13/var_model/files/sub-1_ts-innov.txt")

#results from diff normalised laplacian
taus = io.load_txt("data/test13/norm_diffusion_model/files/sub-1_tau_scalar.tsv")
t0 = taus[0] #self loops tau
t1 = taus[1]
E_diff = io.load_txt("data/test13/norm_diffusion_model/files/sub-1_ts-innov.tsv.gz")

#results from diff NOT normalised laplacian
taus_nn = io.load_txt("data/test13/norm_diffusion_model/files/sub-1_tau_scalar.tsv")
t0_nn = taus[0] #self loops tau
t1_nn = taus[1]
E_diff_nn = io.load_txt("data/test13/not_norm_diffusion_model/files/sub-1_ts-innov.tsv.gz")

#calculate the M matrices (I - tao0 I - tau1 L1)
n = conn_mat.shape[0] #shape for later, square matrix
I = np.identity(n)

L1_nn, degree = laplacian.compute_laplacian(conn_mat, selfloops="degree")
L1 = laplacian.normalisation(L1_nn, degree, norm="rwo")


M_nn = I - t0 * I - t1 * L1_nn
M = I - t0 * I - t1 * L1

#####################################################àà
###### CHECK IF A == (I - tao0 I - tau1 L1)
########################################################

fig, a = plt.subplots(4,3, figsize=(10,10), dpi = 300)

# normal, how they are
pos00 = a[0,0].imshow(conn_mat, interpolation='nearest', aspect="auto"); a[0,0].set_title(f"functinal connectivity")
pos10 = a[1,0].imshow(A, interpolation='nearest', aspect="auto"); a[1,0].set_title(f"A, VAR matrix")
pos20 = a[2,0].imshow(M, interpolation='nearest', aspect="auto"); a[2,0].set_title(f"M, similar to VAR matrix")
pos30 = a[3,0].imshow(M_nn, interpolation='nearest', aspect="auto"); a[3,0].set_title(f"M nn, similar to VAR matrix")

plt.colorbar(pos00, ax=a[0,0]) #plot to whcihc the colorbar refer, the plot where the colorbar will be drawn
plt.colorbar(pos10, ax=a[1,0])
plt.colorbar(pos20, ax=a[2,0])
plt.colorbar(pos30, ax=a[3,0])

# normalized overall

def normalize(x): return (x - np.mean(x)) / np.std(x)
print(np.mean(normalize(A)), np.std(normalize(A)))

pos01 = a[0,1].imshow(normalize(conn_mat), interpolation='nearest', aspect="auto"); a[0,1].set_title(f"functinal connectivity")
pos11 = a[1,1].imshow(normalize(A), interpolation='nearest', aspect="auto"); a[1,1].set_title(f"A, VAR matrix")
pos21 = a[2,1].imshow(normalize(M), interpolation='nearest', aspect="auto"); a[2,1].set_title(f"M, similar to VAR matrix")
pos31 = a[3,1].imshow(normalize(M_nn), interpolation='nearest', aspect="auto"); a[3,1].set_title(f"M nn, similar to VAR matrix")

plt.colorbar(pos01, ax=a[0,1])
plt.colorbar(pos11, ax=a[1,1])
plt.colorbar(pos21, ax=a[2,1])
plt.colorbar(pos31, ax=a[3,1])

# clipped to have same range of valuues

pos02 = a[0,2].imshow(conn_mat, interpolation='nearest', aspect="auto", vmin=0, vmax=1); a[0,2].set_title(f"functinal connectivity")
pos12 = a[1,2].imshow(A, interpolation='nearest', aspect="auto", vmin=0, vmax=1); a[1,2].set_title(f"A, VAR matrix")
pos22 = a[2,2].imshow(M, interpolation='nearest', aspect="auto", vmin=0, vmax=1); a[2,2].set_title(f"M, similar to VAR matrix")
pos32 = a[3,2].imshow(M_nn, interpolation='nearest', aspect="auto", vmin=0, vmax=1); a[3,2].set_title(f"M nn, similar to VAR matrix")

plt.colorbar(pos02, ax=a[0,2]) 
plt.colorbar(pos12, ax=a[1,2])
plt.colorbar(pos22, ax=a[2,2])
plt.colorbar(pos32, ax=a[3,2])

# remvoce the thiks form images

for i in range(4): 
    for j in range(3):
        a[i,j].tick_params(left = False, right = False , labelleft = False , labelbottom = False, bottom = False) 

fig.suptitle("Nothing done --- z-scored --- scaled 0-1")
fig.savefig("data/test13/A_M.png")

#####################################################àà
###### PLOT M WITH CONSTRAST
########################################################

fig, a = plt.subplots(1,2,dpi=300)

a[0].imshow(M, interpolation='nearest', aspect="auto", vmin=0, vmax=0.01); a[0].set_title(f"M, similar to VAR matrix")
a[1].imshow(M_nn, interpolation='nearest', aspect="auto", vmin=0, vmax=0.01); a[1].set_title(f"M_nn, similar to VAR matrix")

fig.savefig("data/test13/M_contrast.png")


####################################################
###### PLOT ERRORS
#####################################################à
fig, a = plt.subplots(3,4, figsize = (30, 10), dpi = 300)

# normal, how they are

pos00 = a[0,0].imshow(E_var, interpolation='nearest', aspect="auto"); a[0,0].set_title(f"E VAR")
pos10 = a[1,0].imshow(E_diff, interpolation='nearest', aspect="auto"); a[1,0].set_title(f"E Diffusion")
pos20 = a[2,0].imshow(E_diff_nn, interpolation='nearest', aspect="auto"); a[2,0].set_title(f"E Diffusion nn")

plt.colorbar(pos00, ax=a[0,0]) 
plt.colorbar(pos10, ax=a[1,0])
plt.colorbar(pos20, ax=a[2,0])

# normalised with overall mean and std

from nigsp.operations.timeseries import normalise_ts #glaballi all the 2 axis

pos01 = a[0,1].imshow(normalise_ts(E_var, globally=True), interpolation='nearest', aspect="auto"); a[0,1].set_title(f"E VAR")
pos11 = a[1,1].imshow(normalise_ts(E_diff, globally=True), interpolation='nearest', aspect="auto"); a[1,1].set_title(f"E Diffusion")
pos21 = a[2,1].imshow(normalise_ts(E_diff_nn, globally=True), interpolation='nearest', aspect="auto"); a[2,1].set_title(f"E Diffusion nn")

plt.colorbar(pos01, ax=a[0,1]) 
plt.colorbar(pos11, ax=a[1,1])
plt.colorbar(pos21, ax=a[2,1])

# normalise only in time dimension

pos02 = a[0,2].imshow(normalise_ts(E_var, globally=False), interpolation='nearest', aspect="auto"); a[0,2].set_title(f"E VAR")
pos12 = a[1,2].imshow(normalise_ts(E_diff, globally=False), interpolation='nearest', aspect="auto"); a[1,2].set_title(f"E Diffusion")
pos22 = a[2,2].imshow(normalise_ts(E_diff_nn, globally=False), interpolation='nearest', aspect="auto"); a[2,2].set_title(f"E Diffusion nn")

plt.colorbar(pos02, ax=a[0,2]) 
plt.colorbar(pos12, ax=a[1,2])
plt.colorbar(pos22, ax=a[2,2])

# clipped to have same range of valuues
min = 0; max = 1

pos03 = a[0,3].imshow(E_var, interpolation='nearest', aspect="auto", vmin=min, vmax=max); a[0,3].set_title(f"E VAR")
pos13 = a[1,3].imshow(E_diff, interpolation='nearest', aspect="auto", vmin=min, vmax=max); a[1,3].set_title(f"A, VAR matrix")
pos23 = a[2,3].imshow(E_diff_nn, interpolation='nearest', aspect="auto", vmin=min, vmax=max); a[2,3].set_title(f"M, similar to VAR matrix")

plt.colorbar(pos03, ax=a[0,3]) 
plt.colorbar(pos13, ax=a[1,3])
plt.colorbar(pos23, ax=a[2,3])

# remvoce the thiks form images

for i in range(3): 
    for j in range(4):
        a[i,j].tick_params(left = False, right = False , labelleft = False , labelbottom = False, bottom = False) 

fig.suptitle("Nothing Done --- z-scored --- Normalized across time --- Scaled 0-1")

fig.savefig("data/test13/errors.png")

################################################################
### CALCULATE DIFFERENCE BETWEEN ERRORS
#################################################################

from numpy.linalg import norm

fig, a = plt.subplots(3,2, dpi=300)

a = a.flatten()

pos0 = a[0].imshow(np.abs(E_var - E_diff), interpolation='nearest', aspect="auto"); a[0].set_title(f"abs(E_var - E_diff), norm{norm(np.abs(E_var - E_diff)):.2f}")
pos1 = a[1].imshow(np.abs(E_var - E_diff_nn), interpolation='nearest', aspect="auto"); a[1].set_title(f"abs(E_var - E_diff_nn), norm={norm(np.abs(E_var - E_diff_nn)):.2f}")
pos2 = a[2].imshow(np.abs(E_diff - E_diff_nn), interpolation='nearest', aspect="auto"); a[2].set_title(f"abs(E_diff - E_diff_nn), norm={norm(np.abs(E_diff - E_diff_nn)):.2f}")

pos3 = a[3].imshow(np.abs(E_var - E_diff), interpolation='nearest', aspect="auto", vmin=0, vmax=1); a[3].set_title(f"abs(E_var - E_diff), norm{norm(np.abs(E_var - E_diff)):.2f}")
pos4 = a[4].imshow(np.abs(E_var - E_diff_nn), interpolation='nearest', aspect="auto", vmin=0, vmax=1); a[4].set_title(f"abs(E_var - E_diff_nn), norm={norm(np.abs(E_var - E_diff_nn)):.2f}")
pos5 = a[5].imshow(np.abs(E_diff - E_diff_nn), interpolation='nearest', aspect="auto", vmin=0, vmax=1); a[5].set_title(f"abs(E_diff - E_diff_nn), norm={norm(np.abs(E_diff - E_diff_nn)):.2f}")

plt.colorbar(pos0, ax=a[0]) 
plt.colorbar(pos1, ax=a[1])
plt.colorbar(pos2, ax=a[2])

plt.colorbar(pos3, ax=a[3]) 
plt.colorbar(pos4, ax=a[4])
plt.colorbar(pos5, ax=a[5])


plt.tight_layout()
fig.savefig("data/test13/differences_errors.png")