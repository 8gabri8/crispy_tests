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

"""
    create synthethic ts where tau and strucral matrices are know
    test if the script that uses laplacina as strucutral file is able to predicit the cirrect taus
"""

#set a seed so is the same between laplacins and eigendevompsition
seed = 4 #4 for postive sum
np.random.seed(seed)

path = "data/test21"
if not os.path.exists(f"{path}"):
    os.mkdir(f"{path}")
L_path = "data/test21/power_L"
if not os.path.exists(L_path):
    os.makedirs(L_path)

one_spike_per_row = 1
n_roi=6
n_tps = 50
n = 15 #max power is n-1 (the first matricxi is L^0=I)
    #ex. n=3 L0, L1, L2, L3
n +=1

############################################
### CHOOSE TAUS
############################################

# Create tau_vec as an array of random positive values between 0.02 and 0.1
#ATTENTION IF TAU IS TOO BIG THERE IS A NON CONVERGENCE
#taus = np.random.uniform(low=0.02, high=0.1, size=(n,)) #we want also tau0

#try negative taus
taus = np.random.uniform(low=-0.1, high=0.1, size=(n,)) #we want also tau0

#try tau in ascdending order, so tau0 is the smallest
#taus.sort()

print(taus)

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
### CONSTRUCT LAPLACIANS
############################################

L, degree = laplacian.compute_laplacian(mtx, selfloops="degree")
L = laplacian.normalisation(L, degree, norm="rwo")

io.export_mtx(np.identity(n_roi), f'{L_path}/L_0.mat') #L^0 =  Identity

for i in np.arange(1,n): 
    L_power = np.linalg.matrix_power(L, i)
    #L_power = laplacian.normalisation(L_power, degree, norm="rwo")
    io.export_mtx(L_power,f'{L_path}/L_{i}.mat') 

############################################
### CONSTRUCT SYNTHEIC SIGNAL Yn
############################################

Y = np.zeros((n_roi, n_tps))
Y[:,0] = ts_spike[:,0]

#Yn = [ (1 - tau0) I - tau1 L1 - tau2 L2 - ...] Yn-1 + En
coeff = (1 - taus[0]) * np.identity(n_roi) #can also use L_0.mat

for j in np.arange(1, n): #rest of the summation    
    coeff -= taus[j] * io.load_mat(f'{L_path}/L_{j}.mat')

for i in np.arange(1, n_tps): # foer each timeporint from the second one

    Y[:,i] = coeff @ Y[:, i-1]

    Y[:,i] += ts_spike[:,i] #add spike in this moment

    if yes_noise:
        Y[:,i] += ts_random[:, i]

plot_greyplot(Y,f"{path}/Y_synth.png")
#plt.imshow(Y, cmap="gray", interpolation='none') ; plt.savefig(f"{path}/Y_synth.png")
io.export_mtx(Y, f"{path}/Y_synth.mat")


############################################
### FROM THE SYNTH SINGNAL TRY TO RECONSTRUCT TAUS
############################################

path_struct = [f"{L_path}/L_" + str(i) + ".mat" for i in np.arange(2, n)]
print(path_struct)

crispy_gls_scalar.multitau_gls_estimation(tsfile = f"{path}/Y_synth.mat",
                add_tau0 = True, #--> L^0 = Identity
                structural_files = f"{path}/SC.mat", #strucral matric --> L^1
                structural_files_nonorm = path_struct, #Laplacinasa are already normalised
                sub = "1",
                odr = f"{path}/diffusion_model")


############################################
### COMPARE TAUS
############################################

taus_GT = io.load_mat(f"{path}/taus_GT.mat")
taus_predicted = np.array(io.load_txt(f"{path}/diffusion_model/files/sub-1_tau_scalar.tsv"))
print(taus_GT.shape)
print(taus_predicted.shape)

E = (norm(io.load_txt(f"data/test21/diffusion_model/files/sub-1_ts-innov.tsv.gz")))

fig, a = plt.subplots(1,2,dpi=300, figsize=(10,5))
fig.suptitle(f'N {n_roi}, TS {n_tps}, Taus {n} POWERS OF LAPLACIANS, SUM taus = {np.sum(taus_GT):.2f}', fontsize=16)
a[0].plot(taus_GT, label = "GT Taus")
a[0].set_xlabel("Tau number ...")
a[0].set_ylabel("Tau value")
a[1].plot(taus_predicted, label = f"Predicted Taus, norm(E) = {E:.2f}")
a[1].set_xlabel("Tau number ...")
a[1].set_ylabel("Tau value")

##################
#plot norm of each Laplacina
plot_norm=0
norms = []
for i in range(n):
    norms.append(norm(io.load_mat(f"{L_path}/L_{i}.mat")))
    if i == 0 :
        plt.clf()
        plt.imshow(io.load_mat(f"{L_path}/L_{i}.mat"))
        nL = norm(io.load_mat(f"{L_path}/L_{i}.mat"))
        plt.xlabel(f"{nL}")
        plt.savefig(f"{path}/L_0.png")
norms = np.array(norms)
if plot_norm:
    a[0].plot(norms, label = "Norm(L^i)")
#print(f"NORMS = {norms}")

################à

a[0].legend(loc='upper left')
a[1].legend(loc='upper left')
plt.tight_layout()
plt.savefig(f"{path}/compare_taus.png")


############################################
### y = tau , x = norm
############################################
import matplotlib.cm as cm
colors = cm.rainbow(np.linspace(0, 1, len(taus_predicted)))
colors = np.linspace(0, 1, len(taus_predicted))

# Create a scatter plot with a colorbar
fig, a = plt.subplots(1, 1, dpi=300, figsize=(10, 5))
fig.suptitle(f'N {n_roi}, TS {n_tps}, Taus {n} POWERS OF LAPLACIANS, SUM taus = {np.sum(taus_GT):.2f}', fontsize=16)

# norms = np.concatenate(([norms[-1]], norms[:-1]))
#taus_predicted = np.concatenate(([taus_predicted[-1]], taus_predicted[:-1]))

scatter = a.scatter(norms, taus_predicted, c=colors)#, cmap='rainbow')
#scatter = a.scatter(norms[0], taus_predicted[0])#, c=colors)#, cmap='rainbow')


# Add colorbar
cbar = plt.colorbar(scatter)
cbar.set_label('Tau Predicted')

a.set_xlabel("Norm(L^i)")
a.set_ylabel("Tau value")
plt.xscale('log')
plt.tight_layout()
plt.savefig(f"{path}/norm_tau_with_colorbar.png")
plt.show()

############################################
### ERROR MATRIX, SHOULD SHOW SOME SPIKES
############################################

E = io.load_txt(f"data/test21/diffusion_model/files/sub-1_ts-innov.tsv.gz")

fig, a = plt.subplots(1,1, dpi=300)
a.imshow(E)
plt.savefig(f"{path}/innov_matrix.png")



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