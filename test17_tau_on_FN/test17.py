import numpy as np
from matplotlib import pyplot as plt
from nigsp import io
from crispy import crispy_gls_scalar, crispy_var_model
from nigsp.operations.metrics import functional_connectivity
import os
import scipy.io
import csv
import nigsp
from my_plot_nodes import plot_nodes
import pandas as pd
import seaborn as sns

"""
    divide 360 nodes in the 7 FN. Each node will have one tau (tau1 lesion or tau2 connectome, clauclated at the same time).
	calculate the std, mean in that region 
     and plot on brain
"""

plt.jet()

# what to run the plot_marker()
plots = 0

data_path = "data/test17"
if not os.path.exists(data_path):
    os.makedirs(data_path)

#load strucure matrix
s = io.load_mat('raw/SC_avg56.mat')

# Load lables of the node, in which FN each node belongs
labels = io.load_mat(f'raw/FN_labels.mat')
n_FN = len(np.unique(labels)) #they should be 7 (from 0 to 1)

# indices of nodes in each FN
indices = []
for i in range(n_FN):
    idx= np.where(labels == i)
    indices.append(idx)

#import taus lesion and connectome (clauclated togetehr) (lesion is a lsingle node)
taus_connectome = io.load_mat(f'raw/taus_connectome.mat')
taus_lesion = io.load_mat(f'raw/taus_lesion.mat')

#contains for each FN_i the values of the tau_CONNECTOME 
#list of lists
taus_connectome_by_FN = []
for i in range (n_FN):
    taus_connectome_by_FN.append(taus_connectome[indices[i]]) #take only yhe taus that are in the current FN

taus_lesion_by_FN = []
for i in range (n_FN):
    taus_lesion_by_FN.append(taus_lesion[indices[i]]) #take only yhe taus that are in the current FN

# dataframe
#  taus: 360 lesion | 360conn
#  labels: FN_1, FN_2, ... | FN_1, FN_2
#  les/conn: Lesion, Lesion .... | Connectome, Connnectome, ...
#each coplumn is 360 * 2 = 720
str_labels = [] #isnted of 2 --> FN_2
for i in range(s.shape[0]):
    str_labels.append("FN_" + str(labels[i]))

order = []
for i in range(n_FN):
    order.append("FN_" + str(i))

str_class = []
for i in range(s.shape[0]):
    str_class.append("Lesion")
for i in range(s.shape[0]):
    str_class.append("Connectome")

# print(len(np.concatenate((taus_lesion, taus_connectome)) ))
# print(len(np.concatenate((str_labels, str_labels)) ))
# print(len(str_class ))

data_dict = {
    "taus" : np.concatenate((taus_lesion, taus_connectome)) ,
    "labels" : np.concatenate((str_labels, str_labels)),
    "Lesion/Connectome" : str_class
}

df = pd.DataFrame(data_dict)

###############################
### BOXPLOX
###############################

fig, a = plt.subplots(dpi=300)
plt.grid()

sns.set()
sns.boxplot(ax = a, data=df, x="labels", y="taus", hue="Lesion/Connectome", order=order).set(
    xlabel='Functional Networks', 
    ylabel='Tau')


plt.tight_layout()
plt.savefig(f"{data_path}/boxplot_taus.png")

###############################
### VIOLIN PLOT
###############################

fig, a = plt.subplots(dpi=300)
#plt.grid()

sns.set()
sns.violinplot(ax = a, data=df, x="labels", y="taus", hue="Lesion/Connectome", order=order).set(
    xlabel='Functional Networks', 
    ylabel='Tau')


plt.tight_layout()
plt.savefig(f"{data_path}/violinplot_taus.png")

###############################
### SWARM PLOT
###############################

fig, a = plt.subplots(dpi=300)
#plt.grid()

sns.set()
sns.swarmplot(ax = a, data=df, x="labels", y="taus", hue="Lesion/Connectome", order=order).set(
    xlabel='Functional Networks', 
    ylabel='Tau')


plt.tight_layout()
plt.savefig(f"{data_path}/swarmplot_taus.png")

###############################
### PLOT ON BRAIN
###############################

thr=0.00001
min = np.min(np.concatenate((taus_connectome, taus_lesion)))
max = np.max(np.concatenate((taus_connectome, taus_lesion)))


if plots:
    for FN in range(n_FN):
        print(f"Plotting FN {FN}")
        temp_lesion = np.zeros(s.shape[0])
        temp_connectome = np.zeros(s.shape[0])

        for i in range(s.shape[0]):
            if labels[i] == FN:
                temp_lesion[i] = taus_lesion[i]
                temp_connectome[i] = taus_connectome[i]

        plot_nodes(temp_lesion, atlas=f"raw/atlas.nii.gz", filename=f"data/test17/taus_lesion_FN_{FN}_on_brain.png", thr=thr)
        plot_nodes(temp_connectome, atlas=f"raw/atlas.nii.gz", filename=f"data/test17/taus_connectome_FN_{FN}_on_brain.png", thr=thr)

        plot_nodes(temp_lesion, atlas=f"raw/atlas.nii.gz", filename=f"data/test17/clip_taus_lesion_FN_{FN}_on_brain.png", thr=thr, min=min, max=max)
        plot_nodes(temp_connectome, atlas=f"raw/atlas.nii.gz", filename=f"data/test17/clip_taus_connectome_FN_{FN}_on_brain.png", thr=thr, min=min, max=max)





































####################################
#display information --> TABLE
# k=4
# data = [
#     [""] + ["FN " + str(i) for i in np.arange(1,n_FN+1,1)], #columns title
#     ['stds_lesion'] + [str(round(i,k)) for i in stds_lesion],
#     ['means_lesion'] + [str(round(i,k)) for i in means_lesion],
#     ['stds_connectome'] + [str(round(i,k)) for i in stds_connectome],
#     ['mean_connectome'] + [str(round(i,k)) for i in means_connectome],
# ]

# # Create a figure and axis
# fig, ax = plt.subplots(dpi=300)

# # Hide the axes
# ax.axis('off')

# # Create the table
# table = ax.table(cellText=data, loc='center', cellLoc='center', colLabels=None)

# # Set the style of the table
# table.auto_set_font_size(False)
# table.set_fontsize(5)
# #table.scale(1.2, 1.2)  # Adjust the scale as needed

# plt.savefig("data/test17/table.png")





# dict_connectome = {
#         "FN_0" : taus_connectome[indices[0]],
#         "FN_1" : taus_connectome[indices[1]],
#         "FN_2" : taus_connectome[indices[2]],
#         "FN_3" : taus_connectome[indices[3]],
#         "FN_4" : taus_connectome[indices[4]],
#         "FN_5" : taus_connectome[indices[5]],
#         "FN_6" : taus_connectome[indices[6]],
#         }








# ###############################
# ### BOXPLOX
# ###############################

# # data_dict = {
# #     "taus_connectome" : taus_connectome,
# #     "taus_lesion" : taus_lesion, 
# #     "labels" : labels
# # }

# # df = pd.DataFrame(data_dict)

# # sns.boxplot(data=df, x="class", y="age", hue="alive", fill=False, gap=.1)

# str_labels = ["FN_" + str(i) for i in range(n_FN)]

# fig, a = plt.subplots(1,2, figsize=(15,10), dpi=300)
# a[0].boxplot(taus_lesion_by_FN , labels=str_labels)
# a[1].boxplot(taus_connectome_by_FN, labels=str_labels)

# a[0].set_title("Lesion taus")
# a[1].set_title("Connectome taus")

# plt.savefig(f"{data_path}/boxplot_taus_connectome.png")

# ###############################
# ### VIOLIN PLOT
# ###############################

# def set_axis_style(ax, x_ticks, title):
#     ax.set_xticks(np.arange(1, len(x_ticks) + 1), labels=x_ticks)
#     ax.set_xlim(0.25, len(x_ticks) + 0.75)
#     ax.set_title(title)

# str_labels = ["FN_" + str(i) for i in range(n_FN)]

# fig, a = plt.subplots(1,2, figsize=(20,10), dpi=300)
# a[0].violinplot(taus_lesion_by_FN, showmeans=True)
# a[1].violinplot(taus_connectome_by_FN, showmeans=True)

# set_axis_style(a[0], str_labels, "Lesion taus")
# set_axis_style(a[1], str_labels, "Connectome taus")


# plt.savefig(f"{data_path}/violinplot_taus_connectome.png")