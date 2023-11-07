# import os
import numpy as np
# import scipy
from matplotlib import pyplot as plt
# import gzip
# import pandas as pd
# import copy
from nigsp import io
from crispy import crispy_gls_scalar, crispy_var_model
from nigsp.operations.metrics import functional_connectivity
# from nigsp.viz import plot_connectivity
# from nigsp.operations import laplacian
# from nilearn.plotting import plot_matrix

"""
    check if something change if insted of giving average strucutral matrix we give the specific strucutral matrix
    we expect that there is a smaler error
"""
calculations = False

#create single strucutral connectome and functional ts
struct_56 = io.load_mat('raw/SCs_56.mat') #NB 3D matrix NxNxsubjects
struct_1 = struct_56[:,:,0] #select only fisrt subject
io.export_mtx(struct_1, 'data/test14/SC_1.mat')

functional_56 = io.load_mat('raw/functional_ts_56.mat') #NB 3D matrix NxNxsubjects
functional_1 = functional_56[:,:,0] #select only fisrt subject
#print(functional_56.shape, functional_1.shape)
io.export_mtx(functional_1, 'data/test14/ts_1.mat')

#calculate tau with strucural everage and specific
if calculations:
    crispy_gls_scalar.multitau_gls_estimation(tsfile = "data/test14/ts_1.mat",
                    structural_files = f"raw/SC_avg56.mat",
                    sub = "1",
                    odr = "data/test14/avg_structural")

    crispy_gls_scalar.multitau_gls_estimation(tsfile = "data/test14/ts_1.mat",
                structural_files = f"data/test14/SC_1.mat",
                sub = "1",
                odr = "data/test14/specific_structural")
    
E_avg = io.load_txt("data/test14/avg_structural/files/sub-1_ts-innov.tsv.gz")
E_spec = io.load_txt("data/test14/specific_structural/files/sub-1_ts-innov.tsv.gz")
taus_avg = np.array(io.load_txt("data/test14/avg_structural/files/sub-1_tau_scalar.tsv"))
taus_spec = np.array(io.load_txt("data/test14/specific_structural/files/sub-1_tau_scalar.tsv"))

  
fig = plt.figure(layout="constrained")
fig, ax = plt.subplot_mosaic("CCc;DDd;XYZ")
ax["X"].scatter([0,1],taus_avg); ax["X"].set_title("taus_avg")
ax["Y"].scatter([0,1],taus_spec); ax["Y"].set_title("taus_specific")

ax["C"].imshow(E_avg, interpolation='nearest', aspect="auto"); ax["C"].set_title(f"E_avg"); ax["C"].tick_params(left = False, right = False , labelleft = False , labelbottom = False, bottom = False)
ax["c"].text(0,0,f"norm = {np.linalg.norm(E_avg):.2f} \nmean = {np.mean(E_avg):.2f}"); ax["c"].tick_params(left = False, right = False , labelleft = False , labelbottom = False, bottom = False)
ax["D"].imshow(E_spec, interpolation='nearest', aspect="auto"); ax["D"].set_title(f"E_spec"); ax["D"].tick_params(left = False, right = False , labelleft = False , labelbottom = False, bottom = False)
ax["d"].text(0,0,f"norm = {np.linalg.norm(E_spec):.2f} \nmean = {np.mean(E_spec):.2f}"); ax["d"].tick_params(left = False, right = False , labelleft = False , labelbottom = False, bottom = False)

#ax["Z"].text(0,0,f"E_spec-E_avg\nnorm = {(np.subtract(E_spec, E_avg)):.2f} \nmean = {np.mean(np.subtract(E_spec,E_avg)):.2f}"); ax["Z"].tick_params(left = False, right = False , labelleft = False , labelbottom = False, bottom = False)


plt.tight_layout()
fig.savefig("data/test14/avg_Vs_specific.png")
#identify_axes(ax_dict)
