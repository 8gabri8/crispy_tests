from nigsp import io, viz
import numpy as np
import os
from matplotlib import pyplot as plt
import pandas as pd
import csv
from nigsp.operations.timeseries import resize_ts

""""
make a lesion in the connection between 2 hemisphere (ie matrix with a hole in the lower left part)
for diffenrt positions of the hole, plot tau Vs size of the hole (noth for lesion part and paz)
NB the lesion are NOT weighted, they are just removed
"""

calculations = False
#360/2 = 180/45 = 4 = how many rectanghle in a row
#9*9 = 81 holes
side = 45
n_matrices = np.power((360/2)/side, 2)


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
    print(s.shape)

    #create strucutral matrice of ill region
    n = 0 #how many matrices we have created
    all_lesion = [] #cointains all the lesion matrices
    all_paz = []
    labels = []

    offset = 0
    for yoffset in np.arange(0, 180, side):
       for xoffset in np.arange(180, 360, side):
        rangex = np.arange(xoffset, xoffset + side, 1) #i must starts from the midell half 
        rangey = np.arange(yoffset, yoffset + side, 1)

        print(f"calculation {n} matrix")
        s_lesion = np.zeros_like(s) #bello

        for i in range(s.shape[0]):
            for j in range(s.shape[1]):
                if i in rangex and j in rangey: #AND
                    s_lesion[i, j] = s[i, j]
                    s_lesion[j, i] = s[j, i]
        
        s_paz = s - s_lesion

        io.export_mtx(s_lesion,f'matrices_lesion/s_lesion_{n}.mat')
        io.export_mtx(s_paz,f'matrices_paz/s_paz_{n}.mat')
        all_lesion.append(s_lesion)
        all_paz.append(s_paz)

        labels.append(str(n))
        n+=1
        
    #plot the matrices
    fig, ax = plt.subplots(2, n, figsize=(20,10))
    for i in range(n):
        ax[0,i].matshow(np.log(all_paz[i]), aspect="auto"); ax[0,i].axis('off')
        ax[1,i].matshow(np.log(all_lesion[i]), aspect="auto"); ax[1,i].axis('off')
    #plt.tight_layout()
    fig.subplots_adjust(left=0.1, right=0.9, bottom=0.1, top=0.9, wspace=0.4, hspace=0.001)
    fig.savefig("matrices.png")
    plt.show()

    for x in labels:
        os.system(f"python3 ../crispy_gls_scalar.py -not0 -s matrices_paz/s_paz_{x}.mat -f ../RS_1subj.mat -sub 1 -od paz/paz_{x}")
        os.system(f"python3 ../crispy_gls_scalar.py -not0 -s matrices_lesion/s_lesion_{x}.mat -f ../RS_1subj.mat -sub 1 -od lesion/lesion_{x}")
    os.system(f"python3 ../crispy_gls_scalar.py -not0 -s ../SC_avg56.mat -f ../RS_1subj.mat -sub 1 -od control")

numbers = np.arange(0, n_matrices, 1)
labels = []
for n in numbers:
    labels.append(str(int(n)))

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
#plt.xlabel('size hole(lesion)')
plt.ylabel('tau')
plt.legend()

fig.savefig("barplot.png")
plt.show()


