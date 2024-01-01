from nigsp import io, viz
import numpy as np
import os
from matplotlib import pyplot as plt
import pandas as pd
import csv
from nigsp.operations.timeseries import resize_ts

def display_table(headers, data):
    # Calculate the maximum width for each column
    column_widths = [max(len(str(item)) for item in col) for col in zip(headers, *data)]
    
    # Print table headers
    print("\n")
    print(" | ".join(f"{header:<{width}}" for header, width in zip(headers, column_widths)))
    
    # Print table separator
    print("-" * (sum(column_widths) + len(headers) * 3 - 1))
    
    # Print table rows
    for row in data:
        print(" | ".join(f"{item:<{width}}" for item, width in zip(row, column_widths)))
    print("\n")

""""
verify that the result of "crispy_new_ols_vector" (ie the E) are the same of when we pass single nodes to normal scirpt
"""

calculations = False
noise = True

if calculations:
    #creteas directiories
    if not os.path.exists("matrices"):
     os.mkdir("matrices")

    #create matrices with good extension
    s = np.loadtxt('mtx_full_synth.1D')
    io.export_mtx(s, "mtx_full_synth.mat")
    ts_orig = np.loadtxt('ts_orig_full_notau0_synth.1D') #fMRI time series WITH NOISE
    io.export_mtx(ts_orig, "ts_orig_full_notau0_synth.mat")
    ts_clean = np.loadtxt('ts_clean_full_notau0_synth.1D') #fMRI time series WITHOUT NOISE
    io.export_mtx(ts_clean, "ts_clean_full_notau0_synth.mat")
    E = np.loadtxt('ts_spike_synth.1D')
    io.export_mtx(E, "ts_spike_synth.mat")

    for i in range(s.shape[0]):
        mat = np.zeros([int(s.shape[0]),int(s.shape[0])])
        mat[:,i] = s[:,i]/2 #i-th colums #-----------> REMEMBER /2
        mat[i,:] = s[i,:]/2
        io.export_mtx(mat,f'matrices/node_{i}.mat')
        plt.matshow(mat)
        plt.show(block=False)
        plt.pause(0.01)

    if noise:
        # execute with a single matrix per node
        os.system("python3 ../crispy_gls_scalar.py -not0 -s matrices/* -f ts_orig_full_notau0_synth.mat -sub 1 -od single_node")
        # execute with the total adjacency matrix
        os.system("python3 ../crispy_gls_scalar.py -not0 -s mtx_full_synth.mat -f ts_orig_full_notau0_synth.mat -sub 1 -od total_adjacency")
        # execute with the other script
        os.system("python3 crispy_ols_vector.py -not0 -s mtx_full_synth.mat -f ts_orig_full_notau0_synth.mat -sub 1 -od ols_vector")
    else:
        # execute with a single matrix per node
        os.system("python3 ../crispy_gls_scalar.py -not0 -s matrices/* -f ts_clean_full_notau0_synth.mat -sub 1 -od single_node_clean")
        # execute with the total adjacency matrix
        os.system("python3 ../crispy_gls_scalar.py -not0 -s mtx_full_synth.mat -f ts_clean_full_notau0_synth.mat -sub 1 -od total_adjacency_clean")
         # execute with the other script
        os.system("python3 crispy_ols_vector.py -not0 -s mtx_full_synth.mat -f ts_clean_full_notau0_synth.mat -sub 1 -od ols_vector_clean")

#if the 2 methos are the same, E1 == E2 == E3
#NB the E from "ts_spike_synth.1D" is the gorund truth, form here the ts were created
E_single = io.load_txt("single_node/files/sub-1_ts-innov.tsv.gz")
E_total = io.load_txt("total_adjacency/files/sub-1_ts-innov.tsv.gz")
E_ols_vector = io.load_txt("ols_vector/files/sub-1_ts-innov.tsv.gz")

#plot E, Ydiff and E-Ydiff
#fig, axes = plt.subplots(2, 3, figsize=(10, 10))
fig, axes = plt.subplots(4, 3, figsize=(10, 10), gridspec_kw={"height_ratios":[1,0.1,1,0.1]})
axes = axes.flatten()
pos_0 = axes[0].imshow(E_single, interpolation='nearest', vmin=-1, vmax=1, aspect='auto'); axes[0].set_title("E_single")
pos_1 = axes[1].imshow(E_total, interpolation='nearest', vmin=-1, vmax=1, aspect='auto'); axes[1].set_title("E_total")
pos_2 = axes[2].imshow(E_ols_vector, interpolation='nearest', vmin=-1, vmax=1, aspect='auto'); axes[2].set_title("E_ols_vector")
pos_6 = axes[6].imshow(np.abs(E_ols_vector - E_single), interpolation='nearest', vmin=0, vmax=1, aspect='auto'); axes[6].set_title("abs(E_ols_vector - E_single)")
pos_7 = axes[7].imshow(np.abs(E_ols_vector - E_total), interpolation='nearest', vmin=0, vmax=1, aspect='auto'); axes[7].set_title("abs(E_ols_vector - E_total)")
pos_8 = axes[8].imshow(np.abs(E_single - E_total), interpolation='nearest', vmin=0, vmax=1, aspect='auto'); axes[8].set_title("abs(E_single - E_total)")

pos_3 = axes[3].set_axis_off()
pos_4 = axes[4]
pos_5 = axes[5].set_axis_off()
axes[3].text(0,0,f"""
STD={np.round(np.std(E_single), 5)}, MEAN={np.round(np.mean(E_single), 5)}\n
STD={np.round(np.std(E_total), 5)}, MEAN={np.round(np.mean(E_total), 5)}\n
STD={np.round(np.std(E_ols_vector), 5)}, MEAN={np.round(np.mean(E_ols_vector), 5)}\n
                    """)
pos_9 = axes[9].set_axis_off(); 
axes[9].text(0, 0,f"""
STD={np.round(np.std(np.abs(E_ols_vector - E_single)), 5)}, MEAN={np.round(np.mean(np.abs(E_ols_vector - E_single)), 5)}, MAX={np.round(np.max(np.abs(E_ols_vector - E_single)), 5)}\n
STD={np.round(np.std(np.abs(E_ols_vector - E_total)), 5)}, MEAN={np.round(np.mean(np.abs(E_ols_vector - E_total)), 5)}, , MAX={np.round(np.max(np.abs(E_ols_vector - E_total)), 5)}\n
STD={np.round(np.std(np.abs(E_single - E_total)), 5)}, MEAN={np.round(np.mean(np.abs(E_single - E_total)), 5)}, , MAX={np.round(np.max(np.abs(E_single - E_total)), 5)}\n
                    """)
pos_10 = axes[10]
pos_11 = axes[11].set_axis_off()

#colorbars
fig.colorbar(pos_0, cax = axes[4], orientation="horizontal")
fig.colorbar(pos_6, cax = axes[10], orientation="horizontal")

#fig.subplots_adjust(left=0.1, right=0.9, bottom=0.1, top=0.9, wspace=0.4, hspace=0.1)
#fig.colorbar(cax = axes[5], ax=np.array(axes[0], axes[3]))#; fig.colorbar(pos_5, ax=axes[3]); fig.colorbar(pos_2, ax=axes[2])
#cbar4 = fig.colorbar(pos_4, ax=axes[5], pad=0.3)
#cbar0 = fig.colorbar(pos_0, ax=axes[5], pad=5)


headers = ["case", "max", "mean"]
data = [
    ["np.abs(E_ols_vector - E_single)", np.max(np.abs(E_ols_vector - E_single)), np.mean(np.abs(E_ols_vector - E_single))],
    ["np.abs(E_ols_vector - E_total)", np.max(np.abs(E_ols_vector - E_total)), np.mean(np.abs(E_ols_vector - E_total))]
]
# print("{:<15} {:<10} {:<20}".format(headers[0], headers[1], headers[2]))
# print("-" * 45)  # Print a line separator
# for row in data:
#     print("{:<15} {:<10} {:<20}".format(row[0], row[1], row[2]))
display_table(headers, data)



plt.tight_layout()
fig.savefig("single_Vs_all.png")
plt.show() 





#     #charge strucutral matrix
#     s = io.load_mat("../SC_avg56.mat") # --> IS SYMMETRIX

#     #create strucutral matrice of ill region
#     cx = np.array(int(s.shape[0]/4)*3)
#     cy = np.array(int(s.shape[1]/4))
#     x = 0
#     y = 0
#     n = 0 #how many matrices we have created
#     all_lesion = [] #cointains all the lesion matrices
#     all_paz = []
#     labels = []

#     while x < (int(s.shape[0]/4)):

#         s_lesion = np.zeros_like(s) #bello

#         rangex = np.arange(cx - x, cx + x)
#         rangey = np.arange(cy - y, cy + y)

#         for i in range(s.shape[0]):
#             for j in range(s.shape[1]):
#                 if i in rangex and j in rangey: #AND
#                     s_lesion[i, j] = s[i, j]
#                     s_lesion[j, i] = s[j, i]
        
#         s_paz = s - s_lesion

#         io.export_mtx(s_lesion,f'matrices_lesion/s_lesion_{x}.mat')
#         io.export_mtx(s_paz,f'matrices_paz/s_paz_{x}.mat')
#         all_lesion.append(s_lesion)
#         all_paz.append(s_paz)

#         labels.append(str(x))
#         x+=5
#         y+=5
#         n+=1
        
#     #plot the matrices
#     fig, ax = plt.subplots(2, n)
#     for i in range(n):
#         ax[0,i].matshow(np.log(all_paz[i])); ax[0,i].axis('off')
#         ax[1,i].matshow(np.log(all_lesion[i])); ax[1,i].axis('off')
#     plt.tight_layout()
#     fig.subplots_adjust(left=0.1, right=0.9, bottom=0.1, top=0.9, wspace=0.4, hspace=0.001)
#     fig.savefig("matrices.png")
#     plt.show()

#     for x in labels:
#         os.system(f"python3 ../crispy_gls_scalar.py -not0 -s matrices_paz/s_paz_{x}.mat -f ../RS_1subj.mat -sub 1 -od paz/paz_{x}")
#         os.system(f"python3 ../crispy_gls_scalar.py -not0 -s matrices_lesion/s_lesion_{x}.mat -f ../RS_1subj.mat -sub 1 -od lesion/lesion_{x}")
#     os.system(f"python3 ../crispy_gls_scalar.py -not0 -s ../SC_avg56.mat -f ../RS_1subj.mat -sub 1 -od control")

# numbers = list(range(0, 86, 5))
# labels = []
# for n in numbers:
#     labels.append(str(n))
# taus_lesion = []
# taus_paz = []
# for label in labels:
#     with open(f"lesion/lesion_{label}/files/sub-1_tau_scalar.tsv", 'r', newline='', encoding='utf-8') as tsvfile:
#         tsv = csv.reader(tsvfile, delimiter='\t')
#         for row in tsv:
#             # Assuming the file contains only one value, extract it from the first row and first column
#             if len(row) > 0:
#                 taus_lesion.append(float(row[0]))
#                 break  # Exit the loop since the value is found
#     with open(f"paz/paz_{label}/files/sub-1_tau_scalar.tsv", 'r', newline='', encoding='utf-8') as tsvfile:
#         tsv = csv.reader(tsvfile, delimiter='\t')
#         for row in tsv:
#             # Assuming the file contains only one value, extract it from the first row and first column
#             if len(row) > 0:
#                 taus_paz.append(float(row[0]))
#                 break  # Exit the loop since the value is found

# print(len(taus_lesion), len(taus_paz))

# cn = np.arange(len(labels))
# print(len(cn))
# width = 0.2

# fig, ax = plt.subplots(1, 1)
# ax.bar(cn - width, taus_lesion, label="lesion", width=width)
# ax.bar(cn + width, taus_paz, label="paz", width=width)
# plt.xticks(cn, labels)
# plt.xlabel('size hole(lesion)')
# plt.ylabel('tau')
# plt.legend()

# fig.savefig("barplot.png")
# plt.show()





