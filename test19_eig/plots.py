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

df = pd.read_csv("data/test19/df.csv")
print(df.head())

fig, a = plt.subplots(1,1,dpi=300, figsize = (5,3))
a.plot(df["0"][:15+1], "o-", label="norm(E)")

a.set_xlabel("Number of components used")
a.ticklabel_format(useOffset=False)
plt.legend(loc="best")
plt.tight_layout()
plt.savefig("data/test19/error.png")
plt.show()
