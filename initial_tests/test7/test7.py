from nigsp import io, viz
import numpy as np
import os
from matplotlib import pyplot as plt
import pandas as pd
import csv

""""
scegliere una regione (sarà la nostra malata). 
Creare la corrispettiva matrice strutturale (solita croce). 
Pesare questa matrice per un set di pesi (ex [0.1:0.1:0.9]) ottenendo così tante matrici strutturali. 
Runnare il codice per ognuna. Plottare in barplot pesi Vs tau.
"""

calcultaions = False
weights = np.arange(0, 1.1, 0.1)

if calcultaions:
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

    all_lesion = [] #cointains all the lesion matrices
    all_paz = []
    labels = [] #weights but in str

    #lesioned region (a part of GM)
    a=130
    b=150
    region = np.array(range(a,b))

    for w in weights:
        s_lesion = np.zeros_like(s)
        n = str(round(w, 2))
        labels.append(n)
        print(labels)

        for i in range(s.shape[0]):
            for j in range(s.shape[1]):
                if i in region or j in region: #AND
                    s_lesion[i, j] = s[i, j] * w
        
        s_paz = s - s_lesion

        io.export_mtx(s_lesion,f'matrices_lesion/s_lesion_{n}.mat')
        io.export_mtx(s_paz,f'matrices_paz/s_paz_{n}.mat')
        all_lesion.append(s_lesion)
        all_paz.append(s_paz)

    
    #plot the matrices
    fig, ax = plt.subplots(2, len(all_lesion), figsize=(10,5))
    for i, w in enumerate(labels):
        ax[0,i].matshow(all_paz[i][a:b, a:b], vmin=-1, vmax=1, aspect='auto'); ax[0,i].axis('off'); ax[0,i].set_title(f'paz_{w}')
         #NB if don't insert vmin/vmax you cannot see the differences
        ax[1,i].matshow(all_lesion[i][a:b, a:b], vmin=-1, vmax=1, aspect='auto'); ax[1,i].axis('off'); ax[1,i].set_title(f'lesion_{w}')
    fig.suptitle("zoom on lesioned region both on paz and lesioned part")
    plt.tight_layout()
    #fig.subplots_adjust(left=0.1, right=0.9, bottom=0.1, top=0.9, wspace=0.4, hspace=0.001)
    fig.savefig("zoom_matrices.png")
    plt.show()

    #plot the matrices
    fig, ax = plt.subplots(2, len(all_lesion), figsize=(15,10))
    for i, w in enumerate(labels):
        ax[0,i].matshow(all_paz[i], vmin=-1, vmax=1, aspect='auto'); ax[0,i].axis('off'); ax[0,i].set_title(f'paz_{w}')
         #NB if don't insert vmin/vmax you cannot see the differences
        ax[1,i].matshow(all_lesion[i], vmin=-1, vmax=1, aspect='auto'); ax[1,i].axis('off'); ax[1,i].set_title(f'lesion_{w}')
    plt.tight_layout()
    #fig.subplots_adjust(left=0.1, right=0.9, bottom=0.1, top=0.9, wspace=0.4, hspace=0.001)
    fig.savefig("matrices.png")
    plt.show()

    for w in labels:
        os.system(f"python3 ../crispy_gls_scalar.py -not0 -s matrices_paz/s_paz_{w}.mat -f ../RS_1subj.mat -sub 1 -od paz/paz_{w}")
        os.system(f"python3 ../crispy_gls_scalar.py -not0 -s matrices_lesion/s_lesion_{w}.mat -f ../RS_1subj.mat -sub 1 -od lesion/lesion_{w}")
    os.system(f"python3 ../crispy_gls_scalar.py -not0 -s ../SC_avg56.mat -f ../RS_1subj.mat -sub 1 -od control")



labels = []
for w in weights:
    labels.append(str(round(w, 2)))

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

cn = np.arange(len(labels))
print(len(cn))
width = 0.2

fig, ax = plt.subplots(1, 1)
ax.bar(cn - width, taus_lesion, label=f"lesion, STD={np.round(np.std(taus_lesion), 2)}, MEAN={np.round(np.mean(taus_lesion), 2)}", width=width)
ax.bar(cn + width, taus_paz, label=f"connectome, STD={np.round(np.std(taus_paz), 4)}, MEAN={np.round(np.mean(taus_paz), 2)}", width=width)

plt.xticks(cn, labels)
plt.xlabel('weight')
plt.ylabel('tau')
ax.legend(loc='lower left', fontsize=15)

fig.savefig("barplot.png")
plt.show()










# weights  = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])
# # d = {}
# # for w in weights:
# #     d[w] = str(w).replace(".", "")

# if create_matrices:
#     #create strucutral matrice of ill region
#     regions = np.array(range(130,150))

#     s = io.load_mat("../SC_avg56.mat")
#     s_few = np.zeros_like(s) #bello
#     for i in range(s.shape[0]):
#         for j in range(s.shape[1]):
#             if i in regions or j in regions: #AND
#                 s_few[i, j] = s[i, j]

#     #create matrices of ill regions scaled with different vlaue

#     if not os.path.exists("matrices"):
#         os.mkdir("matrices")

#     fig, ax = plt.subplots(1, len(weights), figsize=(20,4))

#     for n, w in enumerate(weights):
#         io.export_mtx(s_few * w, f"matrices/M_{d[w]}.mat")
#         ax[n].imshow(s_few * w, interpolation='nearest', aspect='auto', vmin=-1, vmax=1, cmap = "hot") ; ax[n].set_title(f"{w}")
#         ax[n].axis("off")
#         print(f"{w} --> {(s_few * w)[149,150]}")

#     #plt.tight_layout()
#     fig.savefig("all_matrices.png")
#     #plt.show()

#     for w in weights:
#         os.system(f"python3 ../crispy_gls_scalar.py -not0 -s matrices/M_{d[w]}.mat -f ../RS_1subj.mat -sub 1 -od w={d[w]}")
#         print(f"python3 ../crispy_gls_scalar.py -not0 -s M_{d[w]}.mat -f ../RS_1subj.mat -sub 1 -od w={d[w]}")

# taus = np.zeros_like(weights)
# for n, w in enumerate(weights):
#     with open(f"w={d[w]}/files/sub-1_tau_scalar.tsv", 'r', newline='', encoding='utf-8') as tsvfile:
#         tsv = csv.reader(tsvfile, delimiter='\t')
#         for row in tsv:
#             # Assuming the file contains only one value, extract it from the first row and first column
#             if len(row) > 0:
#                 taus[n] = row[0]
#                 break  # Exit the loop since the value is found

# print(taus)
# labels = []
# for w in weights:
#     labels = labels + [str(w)]
# print(labels)
# fig, ax = plt.subplots(1, 1)
# ax.bar(labels, taus, align='center')
# fig.savefig("barplot.png")
# plt.show()

