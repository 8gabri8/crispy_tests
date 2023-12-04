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

path = "data/test20"
if not os.path.exists(f"{path}"):
    os.mkdir(f"{path}")
if not os.path.exists(f"{path}/eigen_struct"):
    os.mkdir(f"{path}/eigen_struct")

# structral  matrix
s = io.load_mat("raw/SC_avg56.mat")

#set number componets
num_components = 3
min = 2 #min number of componet to use

# eigendecomposition
# Compute eigenvalues and eigenvectors
#I KNOW THAT THEY ARE NOT EIGEN..., BUT SINGULAR...
eigenvectors,eigenvalues,  Vt = np.linalg.svd(s)
print(eigenvalues.shape, Vt.shape)

#img = []
#create eigen-strucural matrices

E_eig = []
taus = []

for j in np.arange(min,num_components+1,1): # i want to use frm 2 to numcomp 
    for i in range(j):
        temp = np.outer(eigenvectors[:,i], (eigenvalues[i] * Vt[i].T))
        #img.append(temp)
        io.export_mtx(temp, f"{path}/eigen_struct/{i}.mat")
        print(temp.shape)

    temp = np.zeros(s.shape)
    for i in range(j, s.shape[0]): #the rest of the compntes
        k = np.outer(eigenvectors[i], (eigenvalues[i] * Vt[i].T))
        temp += k
    io.export_mtx(temp, f"{path}/eigen_struct/remaining.mat")

    # fig, a = plt.subplots(2,5, dpi=300)
    # a = a.flatten()
    # for i in range(num_components):
    #     a[i].imshow(img[i], vmin=0, vmax=0.2)

    #plt.show()

    path_struct = [f"{path}/eigen_struct/" + str(i) + ".mat" for i in range(j)]
    path_struct.append(f"{path}/eigen_struct/remaining.mat")
    print(path_struct)

    #without tau 0
    crispy_gls_scalar.multitau_gls_estimation(tsfile = "raw/RS_1subj.mat", 
                                            structural_files = path_struct,
                                            add_tau0 = False, 
                                            sub = "1",
                                            odr = f"{path}/components")


    E_eig.append(norm(io.load_txt(f"{path}/components/files/sub-1_ts-innov.tsv.gz")))
        #NB if in this rounf we use 2 componets, in reality  we pass 3 (+ remaing.mat)
    taus.append(np.array(io.load_txt(f"{path}/components/files/sub-1_tau_scalar.tsv")))

print(taus)
fig, a = plt.subplots(1,2,dpi=300)
for i in range(len(E_eig)):
    a[0].plot(np.arange(1,len(taus[i])+1), taus[i], label=f"with {i+2} components: {E_eig[i]}")

a[1].plot(np.arange(min,len(E_eig)+min), E_eig, "o-", label="Error")
a[1].set_xlabel("Number of components used")
a[1].ticklabel_format(useOffset=False)

plt.legend(loc="best")

plt.tight_layout()
plt.savefig(f"{path}/tuas_E_eig.png")
plt.show()