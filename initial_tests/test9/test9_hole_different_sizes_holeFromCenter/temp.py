from nigsp import io, viz
import numpy as np
import os
from matplotlib import pyplot as plt
import pandas as pd
import csv
from nigsp.operations.timeseries import resize_ts
from nigsp import io, viz


s = io.load_mat("../SC_avg56.mat")
lesion = np.zeros(s.shape)
lesion[:, 30:33] = s[:, 30:33]
lesion[30:33,:] = s[30:33,:]
#plt.imshow(lesion);plt.show()
conn = s - lesion

i = 30
terl = io.load_mat(f"inter_matrices_lesion/s_lesion_{i}.mat")
terc = io.load_mat(f"inter_matrices_paz/s_paz_{i}.mat")

tral = io.load_mat(f"intra_matrices_lesion/s_lesion_{i}.mat")
trac = io.load_mat(f"intra_matrices_paz/s_paz_{i}.mat")

fig, a = plt.subplots(3,2, dpi= 300, figsize=(5,5))
#, aspect="auto"
a[0,0].imshow(np.log(terl), interpolation='nearest', vmin=0, vmax=1); a[0,0].set_title(f"Inter Square Lesion\nsize = {i}"); a[0,0].tick_params(left = False, right = False , labelleft = False , labelbottom = False, bottom = False) 
a[0,1].imshow(np.log(terc), interpolation='nearest', vmin=0, vmax=1); a[0,1].set_title(f"Inter Square Connectome\nsize = {i}"); a[0,1].tick_params(left = False, right = False , labelleft = False , labelbottom = False, bottom = False) 
a[1,0].imshow(np.log(tral), interpolation='nearest', vmin=0, vmax=1); a[1,0].set_title(f"Intra Square Lesion\nsize = {i}"); a[1,0].tick_params(left = False, right = False , labelleft = False , labelbottom = False, bottom = False) 
a[1,1].imshow(np.log(trac), interpolation='nearest', vmin=0, vmax=1); a[1,1].set_title(f"Intra Square Connectome\nsize = {i}"); a[1,1].tick_params(left = False, right = False , labelleft = False , labelbottom = False, bottom = False) 
a[2,0].imshow(np.log(lesion), interpolation='nearest', vmin=0, vmax=0.1); a[2,0].set_title(f"Cross Lesion\nnode = {i}"); a[2,0].tick_params(left = False, right = False , labelleft = False , labelbottom = False, bottom = False) 
a[2,1].imshow(np.log(conn), interpolation='nearest', vmin=0, vmax=0.1); a[2,1].set_title(f"Cross Connectome\nnode = {i}"); a[2,1].tick_params(left = False, right = False , labelleft = False , labelbottom = False, bottom = False) 


plt.tight_layout()
#plt.imshow(m)
plt.savefig("lesions.png")
#plt.show()