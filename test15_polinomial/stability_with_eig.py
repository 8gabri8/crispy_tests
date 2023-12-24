import numpy as np
from matplotlib import pyplot as plt
from nigsp import io
from crispy import crispy_gls_scalar, crispy_var_model
from nigsp.operations.metrics import functional_connectivity
import os
import scipy.io
import csv
import nigsp
# from nigsp.viz import plot_connectivity
from nigsp.operations import laplacian
# from nilearn.plotting import plot_matrix

m = io.load_mat("data/test15/power_L/L_1.mat")

# Compute eigenvalues
eigenvalues = np.linalg.eigvals(m)
print(eigenvalues)

# Plot real part of eigenvalues
plt.bar(range(len(eigenvalues)), np.real(eigenvalues))
plt.xlabel('Real Part')
plt.title('Eigenvalues - Real Part')
plt.grid(True)
plt.show()
plt.savefig("data/test15/stability.png")