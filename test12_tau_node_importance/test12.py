from nigsp import io, viz
import numpy as np
import os
from matplotlib import pyplot as plt
import pandas as pd
import csv
from nigsp.operations.timeseries import resize_ts
from scipy.interpolate import make_interp_spline
from crispy import crispy_gls_scalar, crispy_var_model

"""
    take one node at time --> create lesion and connectome matrix
    for ech node run the script scripsy_sclar
        -s lesion
        -s connectome
        -s lesion connectome --> 2 taus calculated together
    USING -not0 (so without selfloops)

"""

calculations = 0

#creteas directiories
data_path = "data/test12"
if not os.path.exists(data_path):
    os.makedirs(data_path)
if not os.path.exists("data/test12/matrices_lesion"):
    os.mkdir("data/test12/matrices_lesion")
if not os.path.exists("data/test12/matrices_paz"):
    os.mkdir("data/test12/matrices_paz")
if not os.path.exists("data/test12/lesion"):
    os.mkdir("data/test12/lesion")
if not os.path.exists("data/test12/paz"):
    os.mkdir("data/test12/paz")
if not os.path.exists("data/test12/paz-lesion"):
    os.mkdir("data/test12/paz-lesion")

# strctral  matrix
s = io.load_mat("raw/SC_avg56.mat") # --> IS SYMMETRIX

n = s.shape[0]
#all path with matrices of the lesions and paz
string_list_lesion = ["data/test12/matrices_lesion/s_lesion_" + str(i) + ".mat" for i in range(n)]
string_list_paz = ["data/test12/matrices_paz/s_paz_" + str(i) + ".mat" for i in range(n)]


#run the scripts
if calculations:

    #create matrices structrual
    for i in range(360): #one roi at time, starting from the one with lower SDI
        print(f"iteration {i}")
        lesion = np.zeros([int(s.shape[0]),int(s.shape[0])])
        lesion[:,i] = s[:,i] #i-th colums
        lesion[i,:] = s[i,:]
        paz = s - lesion
        io.export_mtx(lesion,f'data/test12/matrices_lesion/s_lesion_{i}.mat') 
        io.export_mtx(paz,f'data/test12/matrices_paz/s_paz_{i}.mat')


    for i in range(360):
        print(f"{i}")

        crispy_gls_scalar.multitau_gls_estimation(tsfile = "raw/RS_1subj.mat",
                structural_files = string_list_paz[i], #f"raw/SC_avg56.mat",
                add_tau0 = False,
                sub = "1",
                odr = f"data/test12/paz/paz_{i}")

        crispy_gls_scalar.multitau_gls_estimation(tsfile = "raw/RS_1subj.mat",
            structural_files = string_list_lesion[i], #f"raw/SC_avg56.mat",
            add_tau0 = False,
            sub = "1",
            odr = f"data/test12/lesion/lesion_{i}")

        crispy_gls_scalar.multitau_gls_estimation(tsfile = "raw/RS_1subj.mat",
            structural_files = [string_list_lesion[i], string_list_paz[i]], #f"raw/SC_avg56.mat",
            add_tau0 = False,
            sub = "1",
            odr = f"data/test12/paz-lesion/paz-lesion_{i}")
    
        #os.system(f"python3 ../crispy_gls_scalar.py -not0 -s matrices_paz/s_paz_{i}.mat -f ../RS_1subj.mat -sub 1 -od paz/paz_{i}")
        #os.system(f"python3 ../crispy_gls_scalar.py -not0 -s matrices_lesion/s_lesion_{i}.mat -f ../RS_1subj.mat -sub 1 -od lesion/lesion_{i}")
        #os.system(f"python3 ../crispy_gls_scalar.py -not0 -s matrices_lesion/s_lesion_{i}.mat matrices_paz/s_paz_{i}.mat -f ../RS_1subj.mat -sub 1 -od paz-lesion/paz-lesion_{i}")



###############################################################################
################# USING DSI
##############################################################################

file_SDI = 'raw/SDI_matlab.tsv'  
SDI = io.load_txt(file_SDI)
SDI = SDI[:,0] #ONLY FIRST SUBJECT
#print(SDI.shape) #should be n_nodesx1 (360)
index = np.argsort(SDI)
#NB index is the array that cointaisn the index to order the SDi array, BUT the nikmber in it are also the ID of the node
#EX index = [5, 6, 360, ...] --> the node in 5 posiotn (so the fitht node) was the one woth SDi min

#extract the data
taus_lesion = []
taus_paz = []
taus_1_paz_lesion = []
taus_2_paz_lesion = []


#NB taus are ORDERD in the array as I take them respecting the order in the index!!!!!
for label in index:
    with open(f"data/test12/lesion/lesion_{label}/files/sub-1_tau_scalar.tsv", 'r', newline='', encoding='utf-8') as tsvfile:
        tsv = csv.reader(tsvfile, delimiter='\t')
        for row in tsv:
            # Assuming the file contains only one value, extract it from the first row and first column
            if len(row) > 0:
                taus_lesion.append(float(row[0]))
                break  # Exit the loop since the value is found

    with open(f"data/test12/paz/paz_{label}/files/sub-1_tau_scalar.tsv", 'r', newline='', encoding='utf-8') as tsvfile:
        tsv = csv.reader(tsvfile, delimiter='\t')
        for row in tsv:
            # Assuming the file contains only one value, extract it from the first row and first column
            if len(row) > 0:
                taus_paz.append(float(row[0]))
                break  # Exit the loop since the value is found
    
    with open(f"data/test12/paz-lesion/paz-lesion_{label}/files/sub-1_tau_scalar.tsv", 'r', newline='', encoding='utf-8') as tsvfile:
        tsv = csv.reader(tsvfile, delimiter='\t')
        k=1 #index of the tau to read
        for row in tsv:
            if len(row) > 0:
                if k==1:
                    taus_1_paz_lesion.append(float(row[0]))
                elif k==2:
                    taus_2_paz_lesion.append(float(row[0]))
            k+=1


#plot
fig, (ax, a) = plt.subplots(1, 2, figsize=(25,10))#, gridspec_kw={'width_ratios': [0.99,0.01]})
cn = np.arange(len(index))
x_ticks = []
for i in index:
    x_ticks.append(f"{i}_{np.round(SDI[i], 3)}")
ax.set_xticks(cn, index, rotation='vertical') #the xticks are the "name"/Id of the node
a.set_xticks(cn, index, rotation='vertical') #the xticks are the "name"/Id of the node


ax.plot(cn, taus_lesion, label=f"lesion, CORR = {np.round(np.corrcoef(taus_lesion, SDI[index])[1,0],2)}")
ax.plot(cn, taus_paz, label=f"connectome, CORR = {np.round(np.corrcoef(taus_paz, SDI[index])[1,0],2)}")
a.plot(cn, taus_1_paz_lesion, label=f"taus_1_lesion, CORR = {np.round(np.corrcoef(taus_1_paz_lesion, SDI[index])[1,0],2)}")
a.plot(cn, taus_2_paz_lesion, label=f"taus_2_connectome, CORR = {np.round(np.corrcoef(taus_2_paz_lesion, SDI[index])[1,0],2)}")

a.plot(np.arange(len(index)), np.interp(SDI[index], (SDI[index].min(), SDI[index].max()), (0, 0.13)), label = "SDI per node (scaled)")
ax.plot(np.arange(len(index)), np.interp(SDI[index], (SDI[index].min(), SDI[index].max()), (0, 0.13)), label = "SDI per node (scaled)")

ax.set_xlabel('nodes ordered by SDI')
ax.set_ylabel('tau')
ax.legend(loc="best", fontsize=15)
ax.title.set_text("lesion connectome separated")
a.title.set_text("lesion connectome together")
a.set_xlabel('nodes ordered by SDI')
a.set_ylabel('tau')
a.legend(loc="best", fontsize=15)
fig.savefig("data/test12/taus_Vs_SDI.png")
plt.show()

################################################################
################ USING STRUCTURAL DEGREE
##################################################################

from nigsp.operations.graph import nodestrength

#node stength
ns = np.abs(nodestrength(s))
index = np.argsort(ns)

taus_lesion = []
taus_paz = []
taus_1_paz_lesion = []
taus_2_paz_lesion = []

#NB taus are ORDERD in the array as I take them respecting the order in the index!!!!!
for label in index:
    with open(f"data/test12/lesion/lesion_{label}/files/sub-1_tau_scalar.tsv", 'r', newline='', encoding='utf-8') as tsvfile:
        tsv = csv.reader(tsvfile, delimiter='\t')
        for row in tsv:
            # Assuming the file contains only one value, extract it from the first row and first column
            if len(row) > 0:
                taus_lesion.append(float(row[0]))
                break  # Exit the loop since the value is found

    with open(f"data/test12/paz/paz_{label}/files/sub-1_tau_scalar.tsv", 'r', newline='', encoding='utf-8') as tsvfile:
        tsv = csv.reader(tsvfile, delimiter='\t')
        for row in tsv:
            # Assuming the file contains only one value, extract it from the first row and first column
            if len(row) > 0:
                taus_paz.append(float(row[0]))
                break  # Exit the loop since the value is found
    
    with open(f"data/test12/paz-lesion/paz-lesion_{label}/files/sub-1_tau_scalar.tsv", 'r', newline='', encoding='utf-8') as tsvfile:
        tsv = csv.reader(tsvfile, delimiter='\t')
        k=1 #index of the tau to read
        for row in tsv:
            if len(row) > 0:
                if k==1:
                    taus_1_paz_lesion.append(float(row[0]))
                elif k==2:
                    taus_2_paz_lesion.append(float(row[0]))
            k+=1

#plot
fig, (ax, a) = plt.subplots(1, 2, figsize=(25,10))#, gridspec_kw={'width_ratios': [0.99,0.01]})
cn = np.arange(len(index))
x_ticks = []
for i in index:
    x_ticks.append(f"{i}")#_{np.round(ns[i], 3)}")
ax.set_xticks(cn, index, rotation='vertical') #the xticks are the "name"/Id of the node
a.set_xticks(cn, index, rotation='vertical')

ax.plot(cn, taus_lesion, label=f"lesion, CORR = {np.round(np.corrcoef(taus_lesion, ns[index])[1,0],2)}")
ax.plot(cn, taus_paz, label=f"connectome, CORR = {np.round(np.corrcoef(taus_paz, ns[index])[1,0],2)}")
a.plot(cn, taus_1_paz_lesion, label=f"taus_1_lesion, CORR = {np.round(np.corrcoef(taus_1_paz_lesion, ns[index])[1,0],2)}")
a.plot(cn, taus_2_paz_lesion, label=f"taus_2_connectome, CORR = {np.round(np.corrcoef(taus_2_paz_lesion, ns[index])[1,0],2)}")

ax.plot(np.arange(len(index)), np.interp(ns[index], (ns[index].min(), ns[index].max()), (0, 0.13)), label = "node_strength_structural (scaled)")
a.plot(np.arange(len(index)), np.interp(ns[index], (ns[index].min(), ns[index].max()), (0, 0.13)), label = "node_strength_structural (scaled)")


ax.set_xlabel('nodes ordered by functional degree')
ax.set_ylabel('tau')
ax.legend(loc="best", fontsize=15)
ax.title.set_text("lesion connectome separated")
a.title.set_text("lesion connectome together")
a.set_xlabel('nodes ordered by functional degree')
a.set_ylabel('tau')
a.legend(loc="best", fontsize=15)
fig.savefig("data/test12/taus_Vs_structural_degree.png")
plt.show()

####################################################################À
############### USING FUNCTIONAL DEGREE
######################################################################

from nigsp.operations.metrics import functional_connectivity

#node stength
ts = io.load_mat("raw/RS_1subj.mat")
fs = np.abs(functional_connectivity(ts)) #matrix where each cell is the correlation of the 2 ts of the 2 nodes
fs = np.sum(np.abs(fs), axis=0) #NB asum of absulte values
index = np.argsort(fs)

taus_lesion = []
taus_paz = []
taus_1_paz_lesion = []
taus_2_paz_lesion = []

#NB taus are ORDERD in the array as I take them respecting the order in the index!!!!!
for label in index:
    with open(f"data/test12/lesion/lesion_{label}/files/sub-1_tau_scalar.tsv", 'r', newline='', encoding='utf-8') as tsvfile:
        tsv = csv.reader(tsvfile, delimiter='\t')
        for row in tsv:
            # Assuming the file contains only one value, extract it from the first row and first column
            if len(row) > 0:
                taus_lesion.append(float(row[0]))
                break  # Exit the loop since the value is found

    with open(f"data/test12/paz/paz_{label}/files/sub-1_tau_scalar.tsv", 'r', newline='', encoding='utf-8') as tsvfile:
        tsv = csv.reader(tsvfile, delimiter='\t')
        for row in tsv:
            # Assuming the file contains only one value, extract it from the first row and first column
            if len(row) > 0:
                taus_paz.append(float(row[0]))
                break  # Exit the loop since the value is found
    
    with open(f"data/test12/paz-lesion/paz-lesion_{label}/files/sub-1_tau_scalar.tsv", 'r', newline='', encoding='utf-8') as tsvfile:
        tsv = csv.reader(tsvfile, delimiter='\t')
        k=1 #index of the tau to read
        for row in tsv:
            if len(row) > 0:
                if k==1:
                    taus_1_paz_lesion.append(float(row[0]))
                elif k==2:
                    taus_2_paz_lesion.append(float(row[0]))
            k+=1

#plot
fig, (ax, a) = plt.subplots(1, 2, figsize=(25,10))#, gridspec_kw={'width_ratios': [0.99,0.01]})
cn = np.arange(len(index))
x_ticks = []
for i in index:
    x_ticks.append(f"{i}_{np.round(fs[i], 3)}")
ax.set_xticks(cn, index, rotation='vertical') #the xticks are the "name"/Id of the node
a.set_xticks(cn, index, rotation='vertical')

ax.plot(cn, taus_lesion, label=f"lesion, CORR = {np.round(np.corrcoef(taus_lesion, fs[index])[1,0],2)}")
ax.plot(cn, taus_paz, label=f"connectome, CORR = {np.round(np.corrcoef(taus_paz, fs[index])[1,0],2)}")
a.plot(cn, taus_1_paz_lesion, label=f"taus_1_lesion, CORR = {np.round(np.corrcoef(taus_1_paz_lesion, fs[index])[1,0],2)}")
a.plot(cn, taus_2_paz_lesion, label=f"taus_2_connectome, CORR = {np.round(np.corrcoef(taus_2_paz_lesion, fs[index])[1,0],2)}")

ax.plot(np.arange(len(index)), np.interp(fs[index], (fs[index].min(), fs[index].max()), (0, 0.13)), label = "node_strength_functional per node (scaled)")
a.plot(np.arange(len(index)), np.interp(fs[index], (fs[index].min(), fs[index].max()), (0, 0.13)), label = "node_strength_functional per node (scaled)")


ax.set_xlabel('nodes ordered by functional degree')
ax.set_ylabel('tau')
ax.legend(loc="best", fontsize=15)
ax.title.set_text("lesion connectome separated")
a.title.set_text("lesion connectome together")
a.set_xlabel('nodes ordered by functional degree')
a.set_ylabel('tau')
a.legend(loc="best", fontsize=15)
fig.savefig("data/test12/taus_Vs_functional_degree.png")
plt.show()


##############################################################
"""
    norm of the E matrix as metrics
    i will have 3 graphs, each one with a diffenrt X axis: 1 where E is from lesiom, 1 form paz, 1 form paz lesion
"""
###################################################################

from numpy.linalg import norm
calculation_2 = 0

#calculate the norm of each matrix
#colaculate the norm for each E matrix, save the norms in a file, load the file, order it argsot, use the indeces to
norms_paz = []
norms_lesion = []
norms_paz_lesion = []

if calculation_2:
    for i in range(s.shape[0]): #for each node/ROI
        E_paz = io.load_txt(f'data/test12/paz/paz_{i}/files/sub-1_ts-innov.tsv.gz') #load the matrix
        norms_paz.append(norm(E_paz))#compute and add the norm

        E_lesion = io.load_txt(f'data/test12/lesion/lesion_{i}/files/sub-1_ts-innov.tsv.gz') #load the matrix
        norms_lesion.append(norm(E_lesion))#compute and add the norm

        E_paz_lesion = io.load_txt(f'data/test12/paz-lesion/paz-lesion_{i}/files/sub-1_ts-innov.tsv.gz') #load the matrix
        norms_paz_lesion.append(norm(E_paz_lesion))#compute and add the norm

        print(f"computing the norm for {i}")

    io.export_mtx(norms_paz,f'data/test12/norms_paz.mat')
    io.export_mtx(norms_lesion,f'data/test12/norms_lesion.mat')
    io.export_mtx(norms_paz_lesion,f'data/test12/norms_paz_lesion.mat')

norms_paz = io.load_mat('data/test12/norms_paz.mat')
norms_lesion = io.load_mat('data/test12/norms_lesion.mat')
norms_paz_lesion = io.load_mat('data/test12/norms_paz_lesion.mat')

index_paz = np.argsort(norms_paz)
index_lesion = np.argsort(norms_lesion)
index_paz_lesion = np.argsort(norms_paz_lesion)

#NB each one has a diffenrt x axis, so diffeerne torfdert to take the taus

taus_lesion = []
for label in index_lesion:
    with open(f"data/test12/lesion/lesion_{label}/files/sub-1_tau_scalar.tsv", 'r', newline='', encoding='utf-8') as tsvfile:
        tsv = csv.reader(tsvfile, delimiter='\t')
        for row in tsv:
            # Assuming the file contains only one value, extract it from the first row and first column
            if len(row) > 0:
                taus_lesion.append(float(row[0]))
                break  # Exit the loop since the value is found

taus_paz = []
for label in index_paz:
    with open(f"data/test12/paz/paz_{label}/files/sub-1_tau_scalar.tsv", 'r', newline='', encoding='utf-8') as tsvfile:
        tsv = csv.reader(tsvfile, delimiter='\t')
        for row in tsv:
            # Assuming the file contains only one value, extract it from the first row and first column
            if len(row) > 0:
                taus_paz.append(float(row[0]))
                break  # Exit the loop since the value is found


taus_1_paz_lesion = []
taus_2_paz_lesion = []
for label in index_paz_lesion:
    with open(f"data/test12/paz-lesion/paz-lesion_{label}/files/sub-1_tau_scalar.tsv", 'r', newline='', encoding='utf-8') as tsvfile:
        tsv = csv.reader(tsvfile, delimiter='\t')
        k=1 #index of the tau to read
        for row in tsv:
            if len(row) > 0:
                if k==1:
                    taus_1_paz_lesion.append(float(row[0]))
                elif k==2:
                    taus_2_paz_lesion.append(float(row[0]))
            k+=1

#plot
fig, (paz, lesion, paz_lesion) = plt.subplots(1, 3, figsize=(25,10))#, gridspec_kw={'width_ratios': [0.99,0.01]})

cn = np.arange(len(index))
x_ticks = []
for i in index:
    x_ticks.append(f"{i}_{np.round(fs[i], 3)}")
paz.set_xticks(cn, index, rotation='vertical') #the xticks are the "name"/Id of the node
lesion.set_xticks(cn, index, rotation='vertical')
paz_lesion.set_xticks(cn, index, rotation='vertical')

#NB tuas_:lesion are in the order of index_lesion, so in the order where the norm of E from leasiojn is increasing
paz.plot(cn, taus_paz, label=f"connectome, CORR = {np.round(np.corrcoef(taus_paz, norms_paz[index_paz])[1,0],2)}")
paz.plot(cn, np.interp(norms_paz[index_paz], (norms_paz[index_paz].min(), norms_paz[index_paz].max()), (0, 0.13)), label="norm(E_paz) value scaled")

lesion.plot(cn, taus_lesion, label=f"lesion, CORR = {np.round(np.corrcoef(taus_lesion, norms_lesion[index_lesion])[1,0],2)}")
lesion.plot(cn, np.interp(norms_lesion[index_lesion], (norms_lesion[index_lesion].min(), norms_lesion[index_lesion].max()), (0, 0.13)), label="norm(E_lesion) value scaled")


paz_lesion.plot(cn, taus_1_paz_lesion, label=f"taus_1_lesion, CORR = {np.round(np.corrcoef(taus_1_paz_lesion, norms_paz_lesion[index_paz_lesion])[1,0],2)}")
paz_lesion.plot(cn, taus_2_paz_lesion, label=f"taus_2_connectome, CORR = {np.round(np.corrcoef(taus_2_paz_lesion, norms_paz_lesion[index_paz_lesion])[1,0],2)}")
paz_lesion.plot(cn, np.interp(norms_paz_lesion[index_paz_lesion], (norms_paz_lesion[index_paz_lesion].min(), norms_paz_lesion[index_paz_lesion].max()), (0, 0.13)), label="norm(E_paz_lesion) value scales")

paz.set_xlabel('nodes ordered by norm(E_connectome)')
paz.set_ylabel('tau')
paz.legend(loc="best", fontsize=15)

lesion.set_xlabel('nodes ordered by norm(E_lesion)')
lesion.set_ylabel('tau')
lesion.legend(loc="best", fontsize=15)

paz_lesion.set_xlabel('nodes ordered by norm(E_lesion_connectome)')
paz_lesion.set_ylabel('tau')
paz_lesion.legend(loc="best", fontsize=15)

fig.savefig("data/test12/taus_Vs_norm.png")
plt.show()

#only paz
fig, a = plt.subplots(2, 2, figsize=(25,10))#, gridspec_kw={'width_ratios': [0.99,0.01]})

cn = np.arange(len(index))
x_ticks = []
for i in index:
    x_ticks.append(f"{i}_{np.round(fs[i], 3)}")
a[0,0].set_xticks(cn, index, rotation='vertical') #the xticks are the "name"/Id of the node
a[0,1].set_xticks(cn, index, rotation='vertical')
a[1,0].set_xticks(cn, index, rotation='vertical')
a[1,1].set_xticks(cn, index, rotation='vertical')



a[0,0].plot(cn, taus_paz, label=f"paz, CORR = {np.round(np.corrcoef(taus_paz, norms_paz[index_paz])[1,0],2)}")
a[0,1].plot(cn, (norms_paz[index_paz] - np.mean(norms_paz[index_paz]))/ np.std(norms_paz[index_paz]), label="norm(E_connectome) demeaned")
a[1,0].plot(cn, taus_lesion, label=f"lesion, CORR = {np.round(np.corrcoef(taus_lesion, norms_lesion[index_lesion])[1,0],2)}")
a[1,1].plot(cn, (norms_lesion[index_lesion] - np.mean(norms_lesion[index_lesion])) / np.std(norms_lesion[index_lesion]), label="norm(E_lesion) demeaned")


a[0,0].set_xlabel('nodes ordered by norm(E_connectome)')
a[0,0].set_ylabel('tau')
a[0,0].legend(loc="best", fontsize=15)
a[0,1].set_xlabel('nodes oderdered by value norm(E_connectome)')
a[0,1].set_ylabel(' value norm(E_pat)')
a[0,1].legend(loc="best", fontsize=15)
a[1,0].set_xlabel('nodes ordered by norm(E_lesion)')
a[1,0].set_ylabel('tau')
a[1,0].legend(loc="best")
a[1,1].set_xlabel('nodes oderdered by value norm(E_lesion)')
a[1,1].set_ylabel(' value norm(E_lesion)')
a[1,1].legend(loc="best", fontsize=15)

fig.savefig("data/test12/norms_separated.png")
plt.tight_layout()
plt.show()

###### sum of tau pazz tau elsf
fig, a = plt.subplots(1,2)
cn = np.arange(len(index))
x_ticks = []
for i in index:
    x_ticks.append(f"{i}_{np.round(fs[i], 3)}")
a[0].set_xticks(cn, index, rotation='vertical') #the xticks are the "name"/Id of the node
a[0].set_xticks(cn, index, rotation='vertical')
a[0].set_xticks(cn, index, rotation='vertical')

a[0].plot(cn, taus_1_paz_lesion, label=f"taus_1_paz_lesion, CORR = {np.round(np.corrcoef(taus_1_paz_lesion, norms_paz_lesion[index_paz_lesion])[1,0],2)}")
a[0].plot(cn, taus_2_paz_lesion, label=f"taus_2_paz_lesion, CORR = {np.round(np.corrcoef(taus_2_paz_lesion, norms_paz_lesion[index_paz_lesion])[1,0],2)}")

#a[1].plot(cn, np.sum(np.array(float(taus_2_paz_lesion)), np.array(float(taus_1_paz_lesion))), label="sum taus")

plt.tight_layout()
fig.savefig("data/test12/sum_taus.png")
plt.show()

##################################################
##############################################################
"""
    taus and norm (on y) Vs node strhngth(structural)
"""

#node stength
ns = np.abs(nodestrength(s))
index = np.argsort(ns)


#order taus by node strrnth
taus_lesion = []
taus_paz = []
taus_1_paz_lesion = []
taus_2_paz_lesion = []

#NB taus are ORDERD in the array as I take them respecting the order in the index!!!!!
for label in index:
    with open(f"data/test12/lesion/lesion_{label}/files/sub-1_tau_scalar.tsv", 'r', newline='', encoding='utf-8') as tsvfile:
        tsv = csv.reader(tsvfile, delimiter='\t')
        for row in tsv:
            # Assuming the file contains only one value, extract it from the first row and first column
            if len(row) > 0:
                taus_lesion.append(float(row[0]))
                break  # Exit the loop since the value is found

    with open(f"data/test12/paz/paz_{label}/files/sub-1_tau_scalar.tsv", 'r', newline='', encoding='utf-8') as tsvfile:
        tsv = csv.reader(tsvfile, delimiter='\t')
        for row in tsv:
            # Assuming the file contains only one value, extract it from the first row and first column
            if len(row) > 0:
                taus_paz.append(float(row[0]))
                break  # Exit the loop since the value is found
    
    with open(f"data/test12/paz-lesion/paz-lesion_{label}/files/sub-1_tau_scalar.tsv", 'r', newline='', encoding='utf-8') as tsvfile:
        tsv = csv.reader(tsvfile, delimiter='\t')
        k=1 #index of the tau to read
        for row in tsv:
            if len(row) > 0:
                if k==1:
                    taus_1_paz_lesion.append(float(row[0]))
                elif k==2:
                    taus_2_paz_lesion.append(float(row[0]))
            k+=1

#oder norm by node stringth
norms_paz = io.load_mat('data/test12/norms_paz.mat')
norms_lesion = io.load_mat('data/test12/norms_lesion.mat')
norms_paz_lesion = io.load_mat('data/test12/norms_paz_lesion.mat')

norms_paz = norms_paz[index]
norms_lesion = norms_lesion[index]
norms_paz_lesion = norms_paz_lesion[index]

#plot
fig, (ax, a, b) = plt.subplots(3,1, figsize=(40,20))#, gridspec_kw={'width_ratios': [0.99,0.01]})
cn = np.arange(len(index))
x_ticks = []
for i in index:
    x_ticks.append(f"{i}")#_{np.round(ns[i], 3)}")

ax.set_xticks(cn, index, rotation='vertical') #the xticks are the "name"/Id of the node
a.set_xticks(cn, index, rotation='vertical')
b.set_xticks(cn, index, rotation='vertical')


ax.plot(cn, taus_lesion, label=f"lesion, CORR = {np.round(np.corrcoef(taus_lesion, ns[index])[1,0],2)}")
ax.plot(cn, taus_paz, label=f"paz, CORR = {np.round(np.corrcoef(taus_paz, ns[index])[1,0],2)}")

b.plot(cn, np.interp(norms_paz, (norms_paz.min(), norms_paz.max()), (0, 0.13)), label=f"norm(E_paz) value scaled, CORR = {np.round(np.corrcoef(norms_paz, ns[index])[1,0],2)}")
b.plot(cn, np.interp(norms_lesion, (norms_lesion.min(), norms_lesion.max()), (0, 0.13)), label=f"norm(E_lesion) value scaled, CORR = {np.round(np.corrcoef(norms_lesion, ns[index])[1,0],2)}")
b.plot(cn, np.interp(norms_paz_lesion, (norms_paz_lesion.min(), norms_paz_lesion.max()), (0, 0.13)), label=f"norm(E_paz_lesion) value scaled, CORR = {np.round(np.corrcoef(norms_paz_lesion, ns[index])[1,0],2)}")

a.plot(cn, taus_1_paz_lesion, label=f"taus_1_paz_lesion, CORR = {np.round(np.corrcoef(taus_1_paz_lesion, ns[index])[1,0],2)}")
a.plot(cn, taus_2_paz_lesion, label=f"taus_2_paz_lesion, CORR = {np.round(np.corrcoef(taus_2_paz_lesion, ns[index])[1,0],2)}")
# a.plot(cn, np.interp(norms_paz, (norms_paz.min(), norms_paz.max()), (0, 0.13)), label=f"norm(E_paz) value scaled, CORR = {np.round(np.corrcoef(norms_paz, ns[index])[1,0],2)}")
# a.plot(cn, np.interp(norms_lesion, (norms_lesion.min(), norms_lesion.max()), (0, 0.13)), label=f"norm(E_lesion) value scaled, CORR = {np.round(np.corrcoef(norms_lesion, ns[index])[1,0],2)}")
# a.plot(cn, np.interp(norms_paz_lesion, (norms_paz_lesion.min(), norms_paz_lesion.max()), (0, 0.13)), label=f"norm(E_paz_lesion) value scaled, CORR = {np.round(np.corrcoef(norms_paz_lesion, ns[index])[1,0],2)}")

#value of ns, it hsudl be increasing monotonically
ax.plot(np.arange(len(index)), np.interp(ns[index], (ns[index].min(), ns[index].max()), (0, 0.13)), label = "node_strength_structural (scaled)")
a.plot(np.arange(len(index)), np.interp(ns[index], (ns[index].min(), ns[index].max()), (0, 0.13)), label = "node_strength_structural (scaled)")
b.plot(np.arange(len(index)), np.interp(ns[index], (ns[index].min(), ns[index].max()), (0, 0.13)), label = "node_strength_structural (scaled)")


ax.set_xlabel('nodes ordered by ns')
ax.set_ylabel('tau')
ax.legend(bbox_to_anchor=(1, 1.0), loc='upper left')
ax.title.set_text("les paz separated")

a.title.set_text("les paz together")
a.set_xlabel('nodes ordered by ns')
a.set_ylabel('tau')
a.legend(bbox_to_anchor=(1, 1.0), loc='upper left')

b.title.set_text("noms(E)")
b.set_xlabel('nodes ordered by ns')
b.set_ylabel('noms(E)')
b.legend(bbox_to_anchor=(1, 1.0), loc='upper left')

fig.savefig("data/test12/taus_norms_Vs_structural_degree.png")
plt.tight_layout()
plt.show()

####

fig, ax  = plt.subplots(1,1, figsize=(20,10))
t1 = np.array(taus_1_paz_lesion) 
t1 = t1 - float(np.mean(t1))
t2 = np.array(taus_2_paz_lesion) 
t2 = t2 - float(np.mean(t2))
t = t1+t2

ax.plot(cn, t1, label=f"taus_1_paz_lesion, CORR = {np.round(np.corrcoef(taus_1_paz_lesion, ns[index])[1,0],2)}")
ax.plot(cn, t2, label=f"taus_2_paz_lesion, CORR = {np.round(np.corrcoef(taus_2_paz_lesion, ns[index])[1,0],2)}")
ax.plot(cn, t, label=f"taus2-taus1 demeaned")
ax.title.set_text("les paz together")
ax.set_xlabel('nodes ordered by ns')
ax.set_ylabel('tau')
ax.legend(bbox_to_anchor=(1, 1.0), loc='upper left')
fig.savefig("data/test12/diff_taus_norms_Vs_structural_degree.png")
plt.tight_layout()
plt.show()
###à



fig, (a,b,c) = plt.subplots(3,1, figsize=(40,10))#, gridspec_kw={'width_ratios': [0.99,0.01]})
a.set_xticks(cn, index, rotation='vertical')
b.set_xticks(cn, index, rotation='vertical')
c.set_xticks(cn, index, rotation='vertical')

a.plot(cn, np.interp(norms_paz, (norms_paz.min(), norms_paz.max()), (0, 0.13)), label=f"norm(E_paz) value scaled, CORR = {np.round(np.corrcoef(norms_paz, ns[index])[1,0],2)}")
b.plot(cn, np.interp(norms_lesion, (norms_lesion.min(), norms_lesion.max()), (0, 0.13)), label=f"norm(E_lesion) value scaled, CORR = {np.round(np.corrcoef(norms_lesion, ns[index])[1,0],2)}")
c.plot(cn, np.interp(norms_paz_lesion, (norms_paz_lesion.min(), norms_paz_lesion.max()), (0, 0.13)), label=f"norm(E_paz_lesion) value scaled, CORR = {np.round(np.corrcoef(norms_paz_lesion, ns[index])[1,0],2)}")

a.title.set_text("norm(E_paz)")
a.set_xlabel('nodes ordered by ns')
a.set_ylabel('norm(E_paz)')
a.legend(bbox_to_anchor=(1, 1.0), loc='upper left')

b.title.set_text("norm(E_lesion)")
b.set_xlabel('nodes ordered by ns')
b.set_ylabel('norm(E_lesion)')
b.legend(bbox_to_anchor=(1, 1.0), loc='upper left')

c.title.set_text("norm(E_paz_lesion)")
c.set_xlabel('nodes ordered by ns')
c.set_ylabel('norm(E_paz_lesion)')
c.legend(bbox_to_anchor=(1, 1.0), loc='upper left')

plt.tight_layout()
fig.savefig("data/test12/norms_Vs_structural_degree.png")
plt.show()

