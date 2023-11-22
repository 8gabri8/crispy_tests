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

# Load lables of the node, in which FN each node belongs
labels = io.load_mat(f'raw/FN_labels.mat')
n_FN = len(np.unique(labels)) #they should be 7 (from 0 to 1)

calculations = 1
if calculations:

    vis_mat = np.zeros((s.shape[0], s.shape[1], n_FN))

    #create strucural matirces, one for each FN
    for FN in range(n_FN):
        temp = np.zeros((s.shape[0], s.shape[1]))
        #print(temp.shape)
        #print(len(range(s.shape[0])), len(data))
        for node, label in zip(range(s.shape[0]), labels):
            if(label == FN):
                print(node, label)
                temp[:,node] = s[:,node] #i-th colums
                temp[node,:] = s[node,:]
        
        temp = temp/2 #ATTENTION!!!!!!!
        io.export_mtx(temp,f'data/test16/FN_{FN}.mat')
        vis_mat[:,:, FN] = temp

    #create one diagonal matric for each FN
    for FN in range(n_FN):
        temp = np.zeros((s.shape[0], s.shape[1]))
        #print(temp.shape)
        #print(len(range(s.shape[0])), len(data))
        for node, label in zip(range(s.shape[0]), labels):
            if(label == FN):
                #print(node, label)
                temp[node,node] = 1 #i-th colums
        
        io.export_mtx(temp,f'data/test16/I_{FN}.mat')
        #vis_mat[:,:, FN] = temp

###########
    #THIS SET THE DIAGONAL??
    # unique_labels = np.unique(labels)
    # #unique_labels = unique_labels[unique_labels>0]
    # mtx = {}
    # for label in unique_labels:
    #     mtx[label] = np.zeros(s.shape)
    #     mtx[label][s==label, s==label] = s[s==label, s==label]
    # fig, a = plt.subplots(1,n_FN, figsize=(10,3), dpi=300)
    # for i in range(n_FN):
    #     a[i].imshow(np.log(mtx[i]), cmap="jet", interpolation='nearest'); a[i].set_title(f"FN {i}")
    # plt.tight_layout()
    # plt.savefig("data/test16/stef_FN_matrices.png")
##############à

    fig, a = plt.subplots(1,2, figsize=(10,3), dpi=300)
    a[0].imshow(s, cmap="jet", interpolation='nearest'); a[0].set_title(f"s")
    a[1].imshow(np.sum(vis_mat, axis=2), cmap="jet", interpolation='nearest'); a[1].set_title(f"sum")
    plt.tight_layout()
    plt.savefig("data/test16/test_structural.png")

    fig, a = plt.subplots(1,n_FN, figsize=(10,3), dpi=300)
    for i in range(n_FN):
        a[i].imshow(np.log(vis_mat[:,:,i]), cmap="jet", interpolation='nearest'); a[i].set_title(f"FN {i}")
    plt.tight_layout()
    plt.savefig("data/test16/FN_matrices.png")

    fig, a = plt.subplots(1,1, figsize=(10,3), dpi=300)
    a.imshow(np.log(vis_mat[:,:,1]), cmap="jet", interpolation='nearest'); a.set_title(f"FN {i}")
    plt.tight_layout()
    plt.savefig("data/test16/single_matrice.png")

    #path of the matrices to use
    string_list = ["data/test16/FN_" + str(i) + ".mat" for i in range(n_FN)]
    #print(string_list)
    I = ["data/test16/I_" + str(i) + ".mat" for i in range(n_FN)]

    crispy_gls_scalar.multitau_gls_estimation(tsfile = "raw/RS_1subj.mat",
                    structural_files = string_list, #f"raw/SC_avg56.mat",
                    add_tau0 = False,
                    bound_taus=True,
                    sub = "1",
                    odr = "data/test16/seven_FN_no_selfloops")
    
    # #WITH SEKF LOOPS, WITH IDENITY AND WITHOUT CLIPPING
    # crispy_gls_scalar.multitau_gls_estimation(tsfile = "raw/RS_1subj.mat",
    #             structural_files = string_list + I, #f"raw/SC_avg56.mat",
    #             add_tau0 = True,
    #             bound_taus=False,
    #             sub = "1",
    #             odr = "data/test16/seven_FN_yes_selfloops")
    
    #WITHOUT IDENITY, YES CLIPPING
    crispy_gls_scalar.multitau_gls_estimation(tsfile = "raw/RS_1subj.mat",
                structural_files = string_list, #f"raw/SC_avg56.mat",
                add_tau0 = True,
                bound_taus=True,
                sub = "1",
                odr = "data/test16/seven_FN_yes_selfloops")
    
taus_sl = np.array(io.load_txt("data/test16/seven_FN_yes_selfloops/files/sub-1_tau_scalar.tsv")) #with self loops
taus_nsl = np.array(io.load_txt("data/test16/seven_FN_no_selfloops/files/sub-1_tau_scalar.tsv"))
print(taus_nsl.shape, taus_sl.shape)
print(taus_sl)

#taus_sl --> (tau0), tau1 (refers to FN_0), ..., t
#labels --> 2, 6, 6, 1, 3, ...

# give to each node the tau corrispondet 
#ex. node10 is in the FN_2 so it will get the tau of the structral matrix of FN_2
tau_nodes_sl = np.zeros(s.shape[0])
tau_nodes_nsl = np.zeros(s.shape[0]) #categorical

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

E_sl = norm(io.load_txt(f"data/test16/seven_FN_yes_selfloops/files/sub-1_ts-innov.tsv.gz"))
E_nsl = norm(io.load_txt(f"data/test16/seven_FN_no_selfloops/files/sub-1_ts-innov.tsv.gz"))


fig, a = plt.subplots(1,1,dpi=300)

a.plot(np.arange(0,len(taus_sl), 1), taus_sl, label=f"With self-loops, norm = {norm(E_sl):.2f}")
a.plot(np.arange(1,len(taus_nsl)+1, 1), taus_nsl, label=f"Without self-loops, norm = {norm(E_nsl):.2f}")

ticks = ["tau0"]+ [f"FN_{i}" for i in range(n_FN)]
print(ticks)
a.set_xticks(range(n_FN +1 ))
a.set_xticklabels(ticks) #must add the tau0
a.set_xlabel("Functional Network")
a.set_ylabel("tau")

plt.legend(loc="best")
plt.grid()
plt.savefig("data/test16/taus.png")


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
    plot_nodes(tau_nodes_sl, "raw/atlas.nii.gz", "data/test16/taus_on_brain_sl.png", 
            display_mode="lyrz")#, figure=fig, axes=a[1], title="taus_on_brain_no_sl_cat")
    plot_nodes(tau_nodes_sl, "raw/atlas.nii.gz", "data/test16/taus_on_brain_sl_cat.png", node_cmap=cmap_sl, display_mode="lyrz")#, figure=fig, axes=a[1], title="taus_on_brain_no_sl_cat")

    #plot_nodes(tau0, "raw/atlas.nii.gz", "data/test16/tau0_on_brain.png", display_mode="lyrz")#, figure=fig, axes=a[1], title="taus_on_brain_no_sl_cat")

    plot_nodes(tau_nodes_nsl, "raw/atlas.nii.gz", "data/test16/taus_on_brain_nsl.png", 
            display_mode="lyrz")#, figure=fig, axes=a[1], title="taus_on_brain_no_nsl_cat")
    plot_nodes(tau_nodes_nsl, "raw/atlas.nii.gz", "data/test16/taus_on_brain_nsl_cat.png", node_cmap=cmap_nsl, display_mode="lyrz")#, figure=fig, axes=a[1], title="taus_on_brain_no_nsl_cat")



