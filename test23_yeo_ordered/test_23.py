import scipy.io as sio #Scipy package to load .mat files
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

data_path = "data/test23"
if not os.path.exists(data_path):
    os.makedirs(data_path)

#load strucure matrix
s = io.load_mat('raw/SC_avg56.mat')

m = sio.loadmat('raw/yeo_RS7_Glasser360.mat')
print(m.keys())
print(m["yeoROIs"])
#yeoOrder = np.ravel(sio.loadmat('raw/yeo_RS7_Glasser360.mat')['yeoOrder']-1)
yeoOrder = np.ravel(sio.loadmat('raw/yeo_RS7_Glasser360.mat')['yeoOrder']-1)
print(yeoOrder)
s_reodered=s[:,yeoOrder][yeoOrder,:]

fig, a = plt.subplots(1,2,dpi=300)
a[0].imshow(np.log(s), cmap="jet", interpolation='nearest'); a[1].set_title(f"Structural matrix")
a[1].imshow(np.log(s_reodered), cmap="jet", interpolation='nearest'); a[1].set_title(f"Structural matrix reordered")
plt.savefig(f"{data_path}/s_reordered.png")

n_FN=7
#how many single values there are == hoe many nodes in each FN
yeoROIs = m["yeoROIs"].flatten()
unique_values, counts = np.unique(yeoROIs, return_counts=True)
counts = np.insert(counts, 0, 0)
counts = np.cumsum(counts)
print(counts)

vis_mat = np.zeros((s.shape[0], s.shape[1], n_FN))

#create strucural matirces, one for each FN
for FN in range(n_FN):
    temp = np.zeros((s.shape[0], s.shape[1]))
    for i in range(s.shape[0]):
        for j in range(s.shape[0]):
                if i in np.arange(counts[FN], counts[FN+1]) or j in np.arange(counts[FN], counts[FN+1]):
                    temp[i,j] = s_reodered[i,j] / 2
    temp[counts[FN]:counts[FN+1],counts[FN]:counts[FN+1]] = s_reodered[counts[FN]:counts[FN+1],counts[FN]:counts[FN+1]]

    io.export_mtx(temp,f'{data_path}/FN_{FN}.mat')
    vis_mat[:,:, FN] = temp

fig, a = plt.subplots(1, n_FN, figsize=(10, 3), dpi=300)

for i in range(n_FN):
    a[i].imshow(np.log(vis_mat[:, :, i]), cmap="jet", interpolation="nearest", vmin=0, vmax=0.01)
    a[i].set_title(f"FN {i}")
    a[i].set_xticks([])
    a[i].set_yticks([])

plt.tight_layout()
plt.savefig(f"{data_path}/FN_matrices.png")

string_list = [f"{data_path}/FN_" + str(i) + ".mat" for i in range(n_FN)]

crispy_gls_scalar.multitau_gls_estimation(tsfile = "raw/RS_1subj.mat",
                structural_files = string_list, #f"raw/SC_avg56.mat",
                add_tau0 = True,
                #bound_taus=True,
                sub = "1",
                odr = f"{data_path}/seven_FN_yes_sl")

crispy_gls_scalar.multitau_gls_estimation(tsfile = "raw/RS_1subj.mat",
                structural_files = string_list, #f"raw/SC_avg56.mat",
                add_tau0 = False,
                #bound_taus=True,
                sub = "1",
                odr = f"{data_path}/seven_FN_no_sl")

taus_sl = np.array(io.load_txt(f"{data_path}/seven_FN_yes_sl/files/sub-1_tau_scalar.tsv")) #with self loops
taus_nsl = np.array(io.load_txt(f"{data_path}/seven_FN_no_sl/files/sub-1_tau_scalar.tsv"))
print(taus_nsl.shape, taus_sl.shape)
print(taus_sl)

#taus_sl --> (tau0), tau1 (refers to FN_0), ..., t
#labels --> 2, 6, 6, 1, 3, ...

# give to each node the tau corrispondet 
#ex. node10 is in the FN_2 so it will get the tau of the structral matrix of FN_2
tau_nodes_sl = np.zeros(s.shape[0])
tau_nodes_nsl = np.zeros(s.shape[0]) #categorical
labels = io.load_mat(f'raw/FN_labels.mat')

for FN in range(n_FN):
    for node, label in zip(range(s.shape[0]), labels):
        if(label == FN):
            tau_nodes_sl[node] = taus_sl[FN+1] # ATTENTION +1, or for example a node in FN_0 will recieve the tau0 )of the identity matrix
            tau_nodes_nsl[node] = taus_nsl[FN] #here the first tau is fereffet to FN:0, no need of summation

#taus_sl = taus_sl[:n_FN] #I don't wna tot take the one of the Indenituts

tau0 = np.ones(s.shape[0]) * taus_sl[0] #all the node have tau0

print(tau_nodes_sl.shape, np.unique(tau_nodes_sl).shape) #360 nodes, 7 differtn taus (not added tau0 in this list)
print(np.unique(tau_nodes_sl))


###########################################################################
### PLOT TAUS
##########################################################################

#def RMSE(c): return np.sqrt(np.square(np.subtract(y_actual,y_predicted)).mean())
from numpy.linalg import norm

E_sl = norm(io.load_txt(f"{data_path}/seven_FN_yes_sl/files/sub-1_ts-innov.tsv.gz"))
E_nsl = norm(io.load_txt(f"{data_path}/seven_FN_no_sl/files/sub-1_ts-innov.tsv.gz"))


fig, a = plt.subplots(1,1,dpi=300,figsize=(5,4))

a.plot(np.arange(0,len(taus_sl), 1), taus_sl, label=f"With self-loops, norm(E) = {norm(E_sl):.2f}")
a.plot(np.arange(1,len(taus_nsl)+1, 1), taus_nsl, label=f"Without self-loops, norm(E) = {norm(E_nsl):.2f}")

ticks = ["tau0"]+ [f"FN_{i}" for i in range(n_FN)]
print(ticks)
a.set_xticks(range(n_FN +1 ))
a.set_xticklabels(ticks) #must add the tau0
a.set_xlabel("Functional Network")
a.set_ylabel("tau")
#plt.suptitle("Structural Matrix Re-Ordered")

plt.legend(loc="best")
plt.grid()
plt.savefig(f"{data_path}/taus.png")

###############
##à WITH RAND STUCTRAL
###############
#load strucure matrix
s = io.load_mat('raw/SC_avg56.mat')


vis_mat = np.zeros((s.shape[0], s.shape[1], n_FN))

counts = [0, 51, 102, 153, 204, 255, 306, 360]

#create strucural matirces, one for each FN
for FN in range(n_FN):
    temp = np.zeros((s.shape[0], s.shape[1]))
    for i in range(s.shape[0]):
        for j in range(s.shape[0]):
                if i in np.arange(counts[FN], counts[FN+1]) or j in np.arange(counts[FN], counts[FN+1]):
                    temp[i,j] = s_reodered[i,j] / 2
    temp[counts[FN]:counts[FN+1],counts[FN]:counts[FN+1]] = s_reodered[counts[FN]:counts[FN+1],counts[FN]:counts[FN+1]]

    io.export_mtx(temp,f'{data_path}/random_{FN}.mat')
    vis_mat[:,:, FN] = temp

fig, a = plt.subplots(1, n_FN, figsize=(10, 3), dpi=300)

for i in range(n_FN):
    a[i].imshow(np.log(vis_mat[:, :, i]), cmap="jet", interpolation="nearest", vmin=0, vmax=1)
    a[i].set_title(f"FN {i}")
    a[i].set_xticks([])
    a[i].set_yticks([])

plt.tight_layout()
plt.savefig(f"{data_path}/random_matrices.png")

string_list = [f"{data_path}/FN_" + str(i) + ".mat" for i in range(n_FN)]
crispy_gls_scalar.multitau_gls_estimation(tsfile = "raw/RS_1subj.mat",
                structural_files = string_list, #f"raw/SC_avg56.mat",
                add_tau0 = True,
                #bound_taus=True,
                sub = "1",
                odr = f"{data_path}/random_FN_yes_sl")

taus_random = np.array(io.load_txt(f"{data_path}/random_FN_yes_sl/files/sub-1_tau_scalar.tsv")) #with self loops
E_random = norm(io.load_txt(f"{data_path}/random_FN_yes_sl/files/sub-1_ts-innov.tsv.gz"))

print(taus_nsl.shape, taus_sl.shape)
print(taus_sl)
print(f"ERRROR ---> {E_random}")
















###########################################################################
### VISUALIZE ON BRAIN
##########################################################################
from my_plot_nodes import plot_nodes
from matplotlib.colors import ListedColormap
#colors = ['red', 'green', 'blue', 'orange', 'purple', 'cyan', 'yellow', 'magenta']
#cmap = ListedColormap(colors)

#create discrete colormapù
unique_values = np.unique(tau_nodes_sl) #should be 8 values (7 FN + tau0)
colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_values)))
cmap_sl = ListedColormap(colors)

unique_values = np.unique(tau_nodes_nsl) #should be 7 values (7 FN)
colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_values)))
cmap_nsl = ListedColormap(colors)

plots=False
if plots:
    #plot 8 taus (one for each FN) in the case where I used self loop in the calculation
    #one coase where the taus have the original value and one where they are made categorical
    plot_nodes(tau_nodes_sl, "raw/atlas.nii.gz", f"{data_path}/taus_on_brain_sl.png", 
            display_mode="lyrz")#, figure=fig, axes=a[1], title="taus_on_brain_no_sl_cat")
    plot_nodes(tau_nodes_sl, "raw/atlas.nii.gz", f"{data_path}/taus_on_brain_sl_cat.png", node_cmap=cmap_sl, display_mode="lyrz")#, figure=fig, axes=a[1], title="taus_on_brain_no_sl_cat")

    #plot_nodes(tau0, "raw/atlas.nii.gz", "data/test16/tau0_on_brain.png", display_mode="lyrz")#, figure=fig, axes=a[1], title="taus_on_brain_no_sl_cat")

    plot_nodes(tau_nodes_nsl, "raw/atlas.nii.gz", f"{data_path}/taus_on_brain_nsl.png", 
            display_mode="lyrz")#, figure=fig, axes=a[1], title="taus_on_brain_no_nsl_cat")
    plot_nodes(tau_nodes_nsl, "raw/atlas.nii.gz", f"{data_path}/taus_on_brain_nsl_cat.png", node_cmap=cmap_nsl, display_mode="lyrz")#, figure=fig, axes=a[1], title="taus_on_brain_no_nsl_cat")

