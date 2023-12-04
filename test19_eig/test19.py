from nigsp import io, viz
import numpy as np
import os
from matplotlib import pyplot as plt
import pandas as pd
import csv
from nigsp.operations.timeseries import resize_ts
from scipy.interpolate import make_interp_spline
from crispy import crispy_gls_scalar, crispy_var_model
from numpy.linalg import norm
from nigsp.operations import laplacian

if not os.path.exists("data/test19"):
    os.mkdir("data/test19")
if not os.path.exists("data/test19/eigen_struct"):
    os.mkdir("data/test19/eigen_struct")

# structral  matrix
s = io.load_mat("raw/SC_avg56.mat")
L, degree = laplacian.compute_laplacian(s, selfloops="degree")
L = laplacian.normalisation(L, degree, norm="rwo")

#set number componets
num_components = 5
min = 2 #min number of componet to use

# eigendecomposition
# Compute eigenvalues and eigenvectors
eigenvalues, eigenvectors = np.linalg.eigh(s)

# Sort eigenvalues and select top k components
sorted_indices = np.argsort(eigenvalues)[::-1]
sorted_eigenvectors = eigenvectors[sorted_indices]
sorted_eigenvalues = eigenvalues[sorted_indices]
print(sorted_eigenvectors.shape)

#create eigen-strucural matrices
E_eig = []
taus = []
img = []

for j in np.arange(min,num_components+1,1): # i want to use frm 2 to numcomp 
    for i in range(j):
        temp = np.outer(sorted_eigenvectors[i], (sorted_eigenvalues[i] * sorted_eigenvectors[i].T))
        img.append(temp)
        io.export_mtx(temp, f"data/test19/eigen_struct/{i}.mat")
        #print(temp.shape)


    temp = np.zeros(s.shape)
    for i in range(j, s.shape[0]): #the rest of the compntes
        k = np.outer(sorted_eigenvectors[i], (sorted_eigenvalues[i] * sorted_eigenvectors[i].T))
        temp += k
    io.export_mtx(temp, f"data/test19/eigen_struct/remaining.mat")

    #plot
    if j == num_components:
        rows = 3
        columns = (num_components + 1) // rows
        fig, a = plt.subplots(rows, columns, dpi=300)
        a = a.flatten()
        for l in range(j):
            a[l].imshow(img[l], vmin=0, vmax=0.01)
        plt.savefig(f"data/test19/eigenbrains.png")

    path_struct = ["data/test19/eigen_struct/" + str(i) + ".mat" for i in range(j)]
    path_struct.append(f"data/test19/eigen_struct/remaining.mat")
    #print(path_struct)

    #without tau 0
    crispy_gls_scalar.multitau_gls_estimation(tsfile = "raw/RS_1subj.mat", 
                                            structural_files = path_struct,
                                            add_tau0 = False, 
                                            sub = "1",
                                            odr = f"data/test19/components")


    E_eig.append(norm(io.load_txt(f"data/test19/components/files/sub-1_ts-innov.tsv.gz")))
        #NB if in this rounf we use 2 componets, in reality  we pass 3 (+ remaing.mat)
    taus.append(np.array(io.load_txt(f"data/test19/components/files/sub-1_tau_scalar.tsv")))

print(taus)
fig, a = plt.subplots(1,2,dpi=300)
for i in range(len(E_eig)):
    a[0].plot(np.arange(1,len(taus[i])+1), taus[i], label=f"with {i+2} components: {E_eig[i]}")

a[1].plot(np.arange(min,len(E_eig)+min), E_eig, "o-", label="Error")
a[1].set_xlabel("Number of components used")
a[1].ticklabel_format(useOffset=False)

plt.legend(loc="best")

plt.tight_layout()
plt.savefig("data/test19/tuas_E_eig.png")
plt.show()