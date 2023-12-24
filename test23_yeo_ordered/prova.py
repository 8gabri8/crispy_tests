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
from nigsp.viz import plot_greyplot
from nigsp.operations import laplacian
from scipy.io import savemat
import scipy.io as sio #Scipy package to load .mat files


m = sio.loadmat('raw/yeo_RS7_Glasser360.mat')
print(m.keys())
print(m["yeoROIs"])
labels = io.load_mat(f'raw/FN_labels.mat')

plt.plot(labels, label="Giulia")
plt.plot(m["yeoROIs"]-1)
plt.legend()
plt.savefig("data/test23/labels.png")