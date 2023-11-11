import numpy as np
from matplotlib import pyplot as plt
from nigsp import io
from crispy import crispy_gls_scalar, crispy_var_model
from nigsp.operations.metrics import functional_connectivity
import os
import scipy.io
import csv
import nigsp


"""
    try to pass 7 structral matrices ot the script (one for each FN), obtain 7 taus --> plot and plot on brain (-s FC1 FC2 ...)
		-try with and without self loop --> very if the error is less
"""

data_path = "data/test16"
if not os.path.exists(data_path):
    os.makedirs(data_path)

#load strucure matrix
s = io.load_mat('raw/SC_avg56.mat')
#print(s.shape)

# Load .mat file
mat = scipy.io.loadmat('raw/Glasser_ch2_yeo_RS7.mat')
#print(mat.keys())
data = mat['yeoROIs'] #data --> array of arraus of single elements, the single lemts are label of in which FN is the specific node
data = data.flatten() #NB 8 FN
#print(np.unique(data))
n = len(np.unique(data))

calculations = 0
if calculations:
    vis_mat = np.zeros((s.shape[0], s.shape[1], n))

    #create strucural matirces, one for each FN
    for FN in np.arange(1,n+1,1):
        temp = np.zeros((s.shape[0], s.shape[1]))
        #print(temp.shape)
        #print(len(range(s.shape[0])), len(data))
        for node, label in zip(range(s.shape[0]), data):
            if(label == FN):
                temp[:,node] = s[:,node] #i-th colums
                temp[node,:] = s[node,:]
        io.export_mtx(temp,f'data/test16/FN_{FN}.mat')
        vis_mat[:,:, FN-1] = temp

    fig, a = plt.subplots(1,n, dpi=300)
    for i in np.arange(1,n+1,1):
        a[i-1].imshow(np.log(vis_mat[:,:,i-1]), interpolation='nearest', aspect="auto"); a[i-1].set_title(f"FN {i}")

    plt.savefig("data/test16/FN_matrices.png")

    #test if they are mutually esclusive
    # temp = s.copy()
    # for i in np.arange(1,7+1,1):
    #     temp = temp - vis_mat[:,:,i-1]
    # if(np.sum(temp) != 0): print("somenthing wrong")

    string_list = ["data/test16/FN_" + str(i) + ".mat" for i in np.arange(1,n+1,1)]
    print(string_list)

    crispy_gls_scalar.multitau_gls_estimation(tsfile = "raw/RS_1subj.mat",
                    structural_files = string_list, #f"raw/SC_avg56.mat",
                    sub = "1",
                    odr = "data/test16/seven_FN_yes_selfloops")
    
    crispy_gls_scalar.multitau_gls_estimation(tsfile = "raw/RS_1subj.mat",
                structural_files = string_list, #f"raw/SC_avg56.mat",
                add_tau0 = False,
                sub = "1",
                odr = "data/test16/seven_FN_no_selfloops")
    
taus_sl = np.array(io.load_txt("data/test16/seven_FN_yes_selfloops/files/sub-1_tau_scalar.tsv")) #with self loops
taus_nsl = np.array(io.load_txt("data/test16/seven_FN_no_selfloops/files/sub-1_tau_scalar.tsv"))
print(taus_nsl.shape, taus_sl.shape)

#taus_sl --> (t0), t1, ..., t8
#data --> 2, 6, 7, 1, 8, ...

# give to each node the tau corrispondet to it's 
tau_nodes_sl = np.zeros(s.shape[0])
tau_nodes_sl_cat = np.zeros(s.shape[0]) #categorical
for FN in np.arange(1,n+1,1):
    for node, label in zip(range(s.shape[0]), data):
        if(label == FN):
            tau_nodes_sl[node] = taus_sl[FN] # NB not -1, i don't want to use tau0
            tau_nodes_sl_cat[node] = FN

tau0 = np.ones(s.shape[0]) * taus_sl[0] #all the node have tau0

print(tau_nodes_sl.shape, np.unique(tau_nodes_sl).shape)
print(np.unique(tau_nodes_sl))

tau_nodes_nsl = np.zeros(s.shape[0])
tau_nodes_nsl_cat = np.zeros(s.shape[0]) #categorical
for FN in np.arange(1,n+1,1):
    for node, label in zip(range(s.shape[0]), data):
        if(label == FN):
            tau_nodes_nsl[node] = taus_nsl[FN-1] # NB yes -1, here we haven't tau0 at the start
            tau_nodes_nsl_cat[node] = FN-1



###########################################################################à
### VISUALIZE
##########################################################################
from my_plot_nodes import plot_nodes
from matplotlib.colors import ListedColormap
colors = ['red', 'green', 'blue', 'orange', 'purple', 'cyan', 'yellow', 'magenta']
cmap = ListedColormap(colors)

#plot 8 taus (one for each FN) in the case where I used self loop in the calculation
#one coase where the taus have the original value and one where they are made categorical
plot_nodes(tau_nodes_sl, "raw/atlas.nii.gz", "data/test16/taus_on_brain_sl.png", 
           display_mode="lyrz")#, figure=fig, axes=a[1], title="taus_on_brain_no_sl_cat")
plot_nodes(tau_nodes_sl_cat, "raw/atlas.nii.gz", "data/test16/taus_on_brain_sl_cat.png", 
           node_cmap=cmap, display_mode="lyrz")#, figure=fig, axes=a[1], title="taus_on_brain_no_sl_cat")

#plot_nodes(tau0, "raw/atlas.nii.gz", "data/test16/tau0_on_brain.png", display_mode="lyrz")#, figure=fig, axes=a[1], title="taus_on_brain_no_sl_cat")

plot_nodes(tau_nodes_nsl, "raw/atlas.nii.gz", "data/test16/taus_on_brain_nsl.png", 
           display_mode="lyrz")#, figure=fig, axes=a[1], title="taus_on_brain_no_nsl_cat")
plot_nodes(tau_nodes_nsl_cat, "raw/atlas.nii.gz", "data/test16/taus_on_brain_nsl_cat.png", 
           node_cmap=cmap, display_mode="lyrz")#, figure=fig, axes=a[1], title="taus_on_brain_no_nsl_cat")



