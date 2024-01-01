from nigsp import io, viz
import numpy as np
import os
from matplotlib import pyplot as plt
import pandas as pd
import csv
from nigsp.operations.timeseries import resize_ts

""""
in test 9, for all sizes the tau is constant? why? maybe the connectiones are too much lower and so there is no effect on the overall diffsuion
let's se if we destroy connection in the same hemipshere (so more weighted) --> maybe the tau will change
make a lesion in the connection in 1 hemisphere (ie matrix with a hole in the upper left part)
for diffenrt sizes of the hole, plot tau Vs size of the hole (noth for lesion part and paz)
NB the lesion are NOT weighted, they are just removed
"""

calculations = False


if calculations:
    #creteas directiories
    if not os.path.exists("matrices_lesion"):
     os.mkdir("matrices_lesion")
    if not os.path.exists("matrices_paz"):
     os.mkdir("matrices_paz")
    if not os.path.exists("lesion"):
     os.mkdir("lesion")
    if not os.path.exists("paz"):
     os.mkdir("paz")

    #charge strucutral matrix
    s = io.load_mat("../SC_avg56.mat") # --> IS SYMMETRIX

    #create strucutral matrice of ill region
    cx = np.array(int(s.shape[0]/2)) #row index
    cy = 0 #columsn index
    x = 0 #side of the lesion along row
    y = 0
    n = 0 #how many matrices we have created
    all_lesion = [] #cointains all the lesion matrices
    all_paz = []
    labels = []

    while cy+y < np.array(int(s.shape[0]/4)):

        s_lesion = np.zeros_like(s) #bello

        rangex = np.arange(cx - x, cx)
        rangey = np.arange(cy, cy + y)

        for i in range(s.shape[0]):
            for j in range(s.shape[1]):
                if i in rangex and j in rangey: #AND
                    s_lesion[i, j] = s[i, j]
                    s_lesion[j, i] = s[j, i]
        
        s_paz = s - s_lesion

        io.export_mtx(s_lesion,f'matrices_lesion/s_lesion_{x}.mat')
        io.export_mtx(s_paz,f'matrices_paz/s_paz_{x}.mat')
        all_lesion.append(s_lesion)
        all_paz.append(s_paz)

        labels.append(str(x))
        x+=5
        y+=5
        n+=1
        
    #plot the matrices
    fig, ax = plt.subplots(2, n, figsize=(15,8))
    for i in range(n):
        ax[0,i].matshow(np.pad(np.log(all_paz[i]), pad_width=5, mode="constant", constant_values=100), aspect="auto"); ax[0,i].axis('off')
        ax[1,i].matshow(np.pad(np.log(all_lesion[i]), pad_width=5, mode="constant", constant_values=100), aspect="auto"); ax[1,i].axis('off')
    plt.tight_layout()
    #fig.subplots_adjust(left=0.1, right=0.9, bottom=0.1, top=0.9, wspace=0.4, hspace=0.001)
    fig.savefig("matrices.png")
    plt.show()

    for x in labels:
        os.system(f"python3 ../crispy_gls_scalar.py -not0 -s matrices_paz/s_paz_{x}.mat -f ../RS_1subj.mat -sub 1 -od paz/paz_{x}")
        os.system(f"python3 ../crispy_gls_scalar.py -not0 -s matrices_lesion/s_lesion_{x}.mat -f ../RS_1subj.mat -sub 1 -od lesion/lesion_{x}")
    os.system(f"python3 ../crispy_gls_scalar.py -not0 -s ../SC_avg56.mat -f ../RS_1subj.mat -sub 1 -od control")

numbers = list(range(0, 86, 5))
labels = []
for n in numbers:
    labels.append(str(n))
taus_lesion = []
taus_paz = []
for label in labels:
    with open(f"lesion/lesion_{label}/files/sub-1_tau_scalar.tsv", 'r', newline='', encoding='utf-8') as tsvfile:
        tsv = csv.reader(tsvfile, delimiter='\t')
        for row in tsv:
            # Assuming the file contains only one value, extract it from the first row and first column
            if len(row) > 0:
                taus_lesion.append(float(row[0]))
                break  # Exit the loop since the value is found
    with open(f"paz/paz_{label}/files/sub-1_tau_scalar.tsv", 'r', newline='', encoding='utf-8') as tsvfile:
        tsv = csv.reader(tsvfile, delimiter='\t')
        for row in tsv:
            # Assuming the file contains only one value, extract it from the first row and first column
            if len(row) > 0:
                taus_paz.append(float(row[0]))
                break  # Exit the loop since the value is found

print(len(taus_lesion), len(taus_paz))

cn = np.arange(len(labels))
print(len(cn))
width = 0.2

fig, ax = plt.subplots(1, 1)
ax.bar(cn - width, taus_lesion, label=f"lesion, STD={np.round(np.std(taus_lesion), 5)}, MEAN={np.round(np.mean(taus_lesion), 5)}", width=width)
ax.bar(cn + width, taus_paz, label=f"paz, STD={np.round(np.std(taus_paz), 5)}, MEAN={np.round(np.mean(taus_paz), 5)}", width=width)
plt.xticks(cn, labels)
plt.xlabel('size hole(lesion)')
plt.ylabel('tau')
#legend_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])  # [x, y, width, height]
fig.legend(bbox_to_anchor=(0.8, 1))

fig.savefig("barplot.png")
plt.show()

# #load E
# E_paz = io.load_txt('paz/files/sub-1_ts-innov.tsv.gz')
# E_lesion = io.load_txt('lesion/files/sub-1_ts-innov.tsv.gz')

# #create Ydiff
# f = io.load_mat('../RS_1subj.mat') #print(f.shape) #-->FUNCTIONAL data
# f = resize_ts(f, resize="norm")
# f = f - f.mean(axis=0)[np.newaxis, ...]
# Ydiff = np.diff(f) #; print(Ydiff.shape)

# #plot E, Ydiff and E-Ydiff
# fig, axes = plt.subplots(2, 1, figsize=(10, 10))
# pos_0 = axes[0].imshow((np.abs(E_paz-Ydiff))[:,:], interpolation='nearest', vmin=0, vmax=0.5); axes[0].set_title("abs(E_paz-Ydiff)")
# pos_1 = axes[1].imshow(np.abs(E_lesion-Ydiff), interpolation='nearest', vmin=0, vmax=0.5); axes[1].set_title("abs(E_lesion-Ydiff)")
# plt.tight_layout()
# fig.savefig("all_Vs_few.png")
# plt.show() 




