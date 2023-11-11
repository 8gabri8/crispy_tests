import numpy as np
from matplotlib import pyplot as plt
from nigsp import io
from crispy import crispy_gls_scalar, crispy_var_model
from nigsp.operations.metrics import functional_connectivity
import os
import scipy.io
import csv
import nigsp

data_path = "data/test17"
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
n_FN = len(np.unique(data))

#use tau of lesion and connectome calculated at the same time
stds_lesion = np.zeros(n_FN) #will contain std of each FN
means_lesion = np.zeros(n_FN)
stds_connectome = np.zeros(n_FN)
means_connectome = np.zeros(n_FN)

for FN in np.arange(1,n_FN+1,1):
    temp_lesion = []
    temp_connectome = []
    taus_lesion = np.zeros(s.shape[0])
    taus_connectome = np.zeros(s.shape[0])
    for node, label in zip(range(s.shape[0]), data):
        if(label == FN):
            taus = np.array(io.load_txt(f"data/test12/paz-lesion/paz-lesion_{node}/files/sub-1_tau_scalar.tsv"))
            temp_lesion.append(taus[0])
            temp_connectome.append(taus[1])
            taus_lesion[node] = taus[0]
            taus_connectome[node] = taus[1]

    stds_lesion[FN-1] = np.std(temp_lesion)
    stds_connectome[FN-1] = np.std(temp_connectome)
    means_lesion[FN-1] = np.mean(temp_lesion)
    means_connectome[FN-1] = np.mean(temp_connectome)

    nigsp.viz.plot_nodes(taus_lesion, f"raw/atlas.nii.gz", "data/test17/taus_lesion_FN_{FN}_on_brain.png", display_mode="lyrz")
    nigsp.vizplot_nodes(taus_connectome, f"raw/atlas.nii.gz", "data/test17/taus_connectome_FN_{FN}_on_brain.png", display_mode="lyrz")


#display information
fig, a = plt.subplots(1,1,dpi=300)
fig.patch.set_visible(False)
a.axis('off')
a.axis('tight')


a.table(cellText=[
    ["stds_lesion"] + str(stds_lesion),
    ["means_lesion"] + str(means_lesion ),
    ["stds_connectome"] + str(stds_connectome),
    ["stds_connectome"] + str(means_connectome),
    ],
    colLabels=["FN " + str(i) for i in np.arange(1,n_FN+1,1)])

plt.savefig("data/test17/table.png")