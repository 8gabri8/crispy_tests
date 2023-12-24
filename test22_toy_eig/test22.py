from nigsp import io, viz
import numpy as np
import os
from matplotlib import pyplot as plt
import pandas as pd
import csv
from nigsp.operations.timeseries import resize_ts
from scipy.interpolate import make_interp_spline
from crispy import crispy_gls_scalar, crispy_var_model
from numpy.linalg import norm
from nigsp.viz import plot_greyplot
from nigsp.operations import laplacian
from scipy.io import savemat

#set a seed so is the same between laplacins and eigendevompsition
seed = 42
np.random.seed(seed)

path = "data/test22"
if not os.path.exists(f"{path}"):
    os.mkdir(f"{path}")
eig_path = "data/test22/eig_brain"
if not os.path.exists(eig_path):
    os.makedirs(eig_path)
L_path = "data/test22/power_L"
if not os.path.exists(L_path):
    os.makedirs(L_path)

one_spike_per_row = 0
n_roi = 6
n_tps = 50
n = 5 #number of single components to use

############################################
### CHOOSE TAUS
############################################

# Create tau_vec as an array of random positive values between 0.02 and 0.1
#ATTENTION IF TAU IS TOO BIG THERE IS A NON CONVERGENCE

#so we need n + 1 + 1(tau0) tus to claiclayte

taus = np.random.uniform(low=0.02, high=0.1, size=(n+2,)) ##we want also tau0
#taus = np.random.uniform(low=-0.1, high=0.1, size=(n+2,)) 


io.export_mtx(taus, f'{path}/taus_GT.mat') 

############################################
### CREATE SC MATRIX
############################################

mtx = np.random.rand(n_roi, n_roi)
mtx = (mtx + mtx.T) / 2
mtx = mtx - mtx.min()
mtx[np.diag_indices(mtx.shape[0])] = 0

io.export_mtx(mtx, f"{path}/SC.mat")

############################################
### CREATE TS
############################################

perc_spike = 0.2 #percentage of spkes to have

ts = np.zeros((n_roi, n_tps)) #ts empty

# will contain where the spike are in the tsù
## Fix spike to be before 2/3 of ts

#one spike for each row
if (one_spike_per_row):
    ts_spike = []
    for i in range(n_roi):
        temp = np.zeros(n_tps)
        k = np.random.randint(0, int((2/3) * n_tps))
        temp[k] = 1
        ts_spike.append(temp)
    ts_spike = np.array(ts_spike, dtype=int) 
else:
    #more spikes per rows
    ts_spike = np.zeros((n_roi, n_tps))
    for i in range(int(n_tps * perc_spike)):
        row = np.random.randint(0, n_roi)
        column = np.random.randint(0, int((2/3) * n_tps))
        ts_spike[row, column] = 1

yes_noise = False
NOISE = 1
if yes_noise:
    NOISE = 0.01
ts_random = np.random.randn(n_roi, n_tps) * NOISE

############################################
### PLOT TS AND SC
############################################

plot_greyplot(ts_spike,f"{path}/ts_spike.png")
plot_greyplot(mtx,f"{path}/mtx_sc_synth.png")

############################################
### CONSTRUCT EIGENBRAINS
############################################


if n > n_roi:
    print("Impossible to have more conponents than ROI")
    import sys
    sys.exit()
#we will have n single compent eigbenbrain + 1 remaning
#so we need n + 1 + 1(tau0) tus to claiclayte

eigenvalues, eigenvectors = np.linalg.eigh(mtx)

# Sort eigenvalues and select top k components
sorted_indices = np.argsort(eigenvalues)[::-1]
sorted_eigenvectors = eigenvectors[sorted_indices]
sorted_eigenvalues = eigenvalues[sorted_indices]

for i in range(n):
    temp = np.outer(sorted_eigenvectors[i], (sorted_eigenvalues[i] * sorted_eigenvectors[i].T))
    io.export_mtx(temp, f"{eig_path}/{i}.mat")
    #print(temp.shape)

temp = np.zeros(mtx.shape)
for i in range(n, mtx.shape[0]): #the rest of the compntes
    k = np.outer(sorted_eigenvectors[i], (sorted_eigenvalues[i] * sorted_eigenvectors[i].T))
    temp += k
io.export_mtx(temp, f"{eig_path}/remaining.mat")

############################################
### CONSTRUCT LAPLACIANS OF EIGENBRAINS
############################################

for i in range(n): 
    L, degree = laplacian.compute_laplacian(io.load_mat(f"{eig_path}/{i}.mat"), selfloops="degree")
    L = laplacian.normalisation(L, degree, norm="rwo")
    io.export_mtx(L,f'{L_path}/L_{i}.mat') 

L, degree = laplacian.compute_laplacian(io.load_mat(f"{eig_path}/remaining.mat"), selfloops="degree")
L = laplacian.normalisation(L, degree, norm="rwo")
io.export_mtx(L,f'{L_path}/L_remaining.mat') 


############################################
### CONSTRUCT SYNTHEIC SIGNAL Yn
############################################

Y = np.zeros((n_roi, n_tps))
Y[:,0] = np.zeros(n_roi)
Ls = [f"{L_path}/L_" +  str(i) + ".mat" for i in range(n)]
Ls.append(f'{L_path}/L_remaining.mat')

#Yn = [ (1 - tau0) I - tau1 L1 - tau2 L2 - ...] Yn-1 + En
for i in np.arange(1, n_tps): # foer each timeporint from the second one
    coeff = (1 - taus[0]) * np.identity(n_roi)

    for num, j in enumerate(Ls): #rest of the summation    
        coeff -= taus[num+1] * io.load_mat(j) #+1 we want to start form tau1

    Y[:,i] = coeff @ Y[:, i-1]

    Y[:,i] += ts_spike[:,i] #add spike in this moment

    if yes_noise:
        Y[:,i] += ts_random[:, i]

plot_greyplot(Y,f"{path}/Y_synth.png")
io.export_mtx(Y, f"{path}/Y_synth.mat")


############################################
### FROM THE SYNTH SINGNAL TRY TO RECONSTRUCT TAUS
############################################

path_struct = [f"{eig_path}/" +  str(i) + ".mat" for i in range(n)]
path_struct.append(f'{eig_path}/remaining.mat')
print(path_struct)

crispy_gls_scalar.multitau_gls_estimation(tsfile = f"{path}/Y_synth.mat",
                add_tau0 = True, #
                structural_files = path_struct, 
                #structural_files_nonorm = path_struct, 
                sub = "1",
                odr = f"{path}/diffusion_model")


############################################
### COMPARE TAUS
############################################

taus_GT = io.load_mat(f"{path}/taus_GT.mat")
taus_predicted = np.array(io.load_txt(f"{path}/diffusion_model/files/sub-1_tau_scalar.tsv"))

E = (norm(io.load_txt(f"data/test22/diffusion_model/files/sub-1_ts-innov.tsv.gz")))


fig, a = plt.subplots(1,2,dpi=300, figsize=(10,5))
fig.suptitle(f'N {n_roi}, TS {n_tps}, Taus {n+2} EIGENDECOMPOSITION', fontsize=16)
a[0].plot(taus_GT, label = "GT")
a[0].set_xlabel("Tau number ...")
a[0].set_ylabel("Tau value")
a[1].plot(taus_predicted, label = f"Predicted, norm {E:.2f}")
a[1].set_xlabel("Tau number ...")
a[1].set_ylabel("Tau value")

a[0].legend(loc='upper left')
a[1].legend(loc='upper left')

plt.tight_layout()
plt.savefig(f"{path}/compare_taus.png")

############################################
### RECONSTRUCT Y WITH PREDICTED TAUS
############################################

# Y_rec = np.zeros((n_roi, n_tps))
# Y_rec[:,0] = np.zeros(n_roi)

# #Yn = [ (1 - tau0) I - tau1 L1 - tau2 L2 - ...] Yn-1 + En
# for i in np.arange(1, n_tps): # foer each timeporint from the second one
#     coeff = (1 - taus_predicted[0]) * np.identity(n_roi)

#     for j in np.arange(1, n): #rest of the summation    
#         coeff -= taus_predicted[j] * io.load_mat(f'{L_path}/L_{j}.mat')

#     Y_rec[:,i] = coeff @ Y_rec[:, i-1]

#     Y_rec[:,i] += ts_spike[:,i] #add spike in this moment

# plot_greyplot(Y_rec,f"{path}/Y_rec_synth.png")
# io.export_mtx(Y_rec, f"{path}/Y_rec_synth.mat")