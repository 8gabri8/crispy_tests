from nigsp import io, viz
import numpy as np
import os
from matplotlib import pyplot as plt
import pandas as pd
import csv
from nigsp.operations.timeseries import resize_ts

""""
buco
    form corner
        inter --> 4 plot for intra
            paz
            lesion
            -s paz lesion
                tau1
                tau2
        intra
            paz
            lesion
            -s paz lesion
                tau1
                tau2

    from center
"""

calculations = True
side = 10 #what add at each iteration of the side of the hole


for region in ["inter", "intra"]:
    if calculations:
        #creteas directiories
        if not os.path.exists(f"{region}_matrices_lesion"):
            os.mkdir(f"{region}_matrices_lesion")
        if not os.path.exists(f"{region}_matrices_paz"):
            os.mkdir(F"{region}_matrices_paz")
        if not os.path.exists(f"{region}_lesion"):
            os.mkdir(f"{region}_lesion")
        if not os.path.exists(f"{region}_paz"):
            os.mkdir(f"{region}_paz")
        if not os.path.exists(f"{region}_paz-lesion"):
            os.mkdir(f"{region}_paz-lesion")
        

        #charge strucutral matrix
        s = io.load_mat("../SC_avg56.mat") # --> IS SYMMETRIX

        #create strucutral matrice of ill region
        if region == "inter":
            cx = np.array(int(s.shape[0]/2))
            cy = 0
        else:
            cx = 0
            cy = 0
        x = 0
        y = 0
        n = 0 #how many matrices we have created
        all_lesion = [] #cointains all the lesion matrices
        all_paz = []
        labels = []

        while x <= (int(s.shape[0]/2)):

            s_lesion = np.zeros_like(s) #bello

            rangex = np.arange(cx, cx + x)
            rangey = np.arange(cy, cy + y)

            for i in range(s.shape[0]):
                for j in range(s.shape[1]):
                    if i in rangex and j in rangey: #AND
                        if region == "inter":
                            s_lesion[i, j] = s[i, j]
                            s_lesion[j, i] = s[j, i]
                        else:
                            s_lesion[i, j] = s[i, j]
                            s_lesion[i+int(s.shape[0]/2), j+int(s.shape[0]/2)] = s[i+int(s.shape[0]/2), j+int(s.shape[0]/2)]
            
            s_paz = s - s_lesion

            io.export_mtx(s_lesion,f'{region}_matrices_lesion/s_lesion_{x}.mat')
            io.export_mtx(s_paz,f'{region}_matrices_paz/s_paz_{x}.mat')
            all_lesion.append(s_lesion)
            all_paz.append(s_paz)

            labels.append(str(x))
            x+=side
            y+=side
            n+=1
            
        #plot the matrices
        fig, ax = plt.subplots(2, n, figsize=(15,8))
        for i in range(n):
            ax[0,i].matshow(np.pad(np.log(all_paz[i]), pad_width=5, mode="constant", constant_values=100), aspect="auto"); ax[0,i].axis('off')
            ax[1,i].matshow(np.pad(np.log(all_lesion[i]), pad_width=5, mode="constant", constant_values=100), aspect="auto"); ax[1,i].axis('off')
        plt.tight_layout()
        #fig.subplots_adjust(left=0.1, right=0.9, bottom=0.1, top=0.9, wspace=0.4, hspace=0.001)
        fig.savefig(f"{region}_matrices.png")
        plt.show()

        for x in labels:
            os.system(f"python3 ../crispy_gls_scalar.py -not0 -s {region}_matrices_paz/s_paz_{x}.mat -f ../RS_1subj.mat -sub 1 -od {region}_paz/paz_{x}")
            os.system(f"python3 ../crispy_gls_scalar.py -not0 -s {region}_matrices_lesion/s_lesion_{x}.mat -f ../RS_1subj.mat -sub 1 -od {region}_lesion/lesion_{x}")
            os.system(f"python3 ../crispy_gls_scalar.py -not0 -s {region}_matrices_lesion/s_lesion_{x}.mat {region}_matrices_paz/s_paz_{x}.mat -f ../RS_1subj.mat -sub 1 -od {region}_paz-lesion/paz-lesion_{x}")       
        os.system(f"python3 ../crispy_gls_scalar.py -not0 -s ../SC_avg56.mat -f ../RS_1subj.mat -sub 1 -od control")


#PLOTTING
numbers = list(range(0, 90+1, side))
labels = [] #how big is the hole
for n in numbers:
    labels.append(str(n))
print(labels)

intra_taus_lesion = []
intra_taus_paz = []
intra_taus_1_paz_lesion = []
intra_taus_2_paz_lesion = []

inter_taus_lesion = []
inter_taus_paz = []
inter_taus_1_paz_lesion = []
inter_taus_2_paz_lesion = []

for region in ["inter", "intra"]:
    for label in labels:
        with open(f"{region}_lesion/lesion_{label}/files/sub-1_tau_scalar.tsv", 'r', newline='', encoding='utf-8') as tsvfile:
            tsv = csv.reader(tsvfile, delimiter='\t')
            for row in tsv:
                # Assuming the file contains only one value, extract it from the first row and first column
                if len(row) > 0:
                    if region == "intra": intra_taus_lesion.append(float(row[0]))
                    else: inter_taus_lesion.append(float(row[0]))
                    break  # Exit the loop since the value is found

        with open(f"{region}_paz/paz_{label}/files/sub-1_tau_scalar.tsv", 'r', newline='', encoding='utf-8') as tsvfile:
            tsv = csv.reader(tsvfile, delimiter='\t')
            for row in tsv:
                # Assuming the file contains only one value, extract it from the first row and first column
                if len(row) > 0:
                    if region == "intra": intra_taus_paz.append(float(row[0]))
                    else: inter_taus_paz.append(float(row[0]))
                    break  # Exit the loop since the value is found
        
        with open(f"{region}_paz-lesion/paz-lesion_{label}/files/sub-1_tau_scalar.tsv", 'r', newline='', encoding='utf-8') as tsvfile:
            tsv = csv.reader(tsvfile, delimiter='\t')
            k=1 #index of the tau to read
            for row in tsv:
                if len(row) > 0:
                    if region == "intra" and k==1:
                        intra_taus_1_paz_lesion.append(float(row[0]))
                    elif region == "intra" and k==2:
                        intra_taus_2_paz_lesion.append(float(row[0]))
                    elif region == "inter" and k==1:
                        inter_taus_1_paz_lesion.append(float(row[0]))
                    elif region == "inter" and k==2:
                        inter_taus_2_paz_lesion.append(float(row[0]))
                k+=1

print(len(inter_taus_lesion), len(inter_taus_paz), len(intra_taus_lesion), len(intra_taus_paz))


fig, (ax, leg) = plt.subplots(1, 2, figsize=(15,20), gridspec_kw={'width_ratios': [0.8,0.2]})
cn = np.arange(len(labels))
ax.set_xticks(cn, labels)

ax.plot(cn, inter_taus_lesion, label=f"inter_lesion, STD={np.round(np.std(inter_taus_lesion), 5)}, MEAN={np.round(np.mean(inter_taus_lesion), 5)}")
ax.plot(cn, inter_taus_paz, label=f"inter_paz, STD={np.round(np.std(inter_taus_paz), 5)}, MEAN={np.round(np.mean(inter_taus_paz), 5)}")
ax.plot(cn, intra_taus_lesion, label=f"intra_lesion, STD={np.round(np.std(intra_taus_lesion), 5)}, MEAN={np.round(np.mean(intra_taus_lesion), 5)}")
ax.plot(cn, intra_taus_paz, label=f"intra_paz, STD={np.round(np.std(intra_taus_paz), 5)}, MEAN={np.round(np.mean(intra_taus_paz), 5)}")
spessore=5
ax.plot(cn, inter_taus_1_paz_lesion, label="inter_taus_1_paz_lesion", lw=spessore)
ax.plot(cn, inter_taus_2_paz_lesion, label="inter_taus_2_paz_lesion", lw=spessore)
ax.plot(cn, intra_taus_1_paz_lesion, label="intra_taus_1_paz_lesion", lw=spessore)
ax.plot(cn, intra_taus_2_paz_lesion, label="intra_taus_2_paz_lesion", lw=spessore)

leg.axis("off")
ax.set_xlabel('side of hole (size of lesion)')
ax.set_ylabel('tau')
ax.legend(loc="best", bbox_to_anchor=(1, 1))
fig.savefig("Holes_from_Center_different_sizes_intra-inter.png")
plt.show()


##########################################

# cn = np.arange(len(labels))
# print(len(cn))
# width = 0.2

# fig, ax = plt.subplots(1, 1)
# ax.bar(cn - width, taus_lesion, label=f"lesion, STD={np.round(np.std(taus_lesion), 5)}, MEAN={np.round(np.mean(taus_lesion), 5)}", width=width)
# ax.bar(cn + width, taus_paz, label=f"paz, STD={np.round(np.std(taus_paz), 5)}, MEAN={np.round(np.mean(taus_paz), 5)}", width=width)
# plt.xticks(cn, labels)
# plt.xlabel('size hole(lesion)')
# plt.ylabel('tau')
# fig.legend(bbox_to_anchor=(0.8, 1))

# fig.savefig("barplot.png")
# plt.show()






