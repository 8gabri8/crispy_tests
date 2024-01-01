from nigsp import io, viz
import numpy as np
import os
from matplotlib import pyplot as plt
import pandas as pd
import csv
from nigsp.operations.timeseries import resize_ts

""""
provare a lesionare la WM, ie creare una matrice con "buchi" simmetrici dove ci sono connessioni 
fra regioni in emisferi diversi (variante del test buco sulla diagonale)
NB ho lesionato il corpo calloso/il collegamento NON uan parte --> quindi oavrò una matrice a blocchi dell aprte lesioanta
"""

create_matrices = False


if create_matrices:
    #create strucutral matrice of ill region
    dx = np.array(range(230,250))
    sx = np.array(range(20,40))

    s = io.load_mat("../SC_avg56.mat") # --> IS SYMMETRIX
    s_few = np.zeros_like(s) #bello

    for i in range(s.shape[0]):
        for j in range(s.shape[1]):
            if i in dx and j in sx: #AND
                s_few[i, j] = s[i, j]
                s_few[j, i] = s[j, i]

    # for i in range(s.shape[0]):
    #     for j in range(s.shape[1]):
    #         if i in dx or j in sx: #AND
    #             s_few[i, j] = s[i, j]
    #             s_few[j, i] = s[j, i]

    s_paz = s - s_few

    fig, ax = plt.subplots(2, 1, figsize=(10, 10))
    ax[0].imshow(np.log(s_few))
    ax[1].imshow(np.log(s_paz))
    fig.savefig("matrices.png")
    plt.show()

    io.export_mtx(s_few,'s_lesioned.mat')
    io.export_mtx(s_paz,'s_paz.mat')

    os.system("python3 ../crispy_gls_scalar.py -not0 -s s_paz.mat -f ../RS_1subj.mat -sub 1 -od paz")
    os.system("python3 ../crispy_gls_scalar.py -not0 -s s_lesioned.mat -f ../RS_1subj.mat -sub 1 -od lesioned")

#load E
E_paz = io.load_txt('paz/files/sub-1_ts-innov.tsv.gz')
E_lesioned = io.load_txt('lesioned/files/sub-1_ts-innov.tsv.gz')

#create Ydiff
f = io.load_mat('../RS_1subj.mat') #print(f.shape) #-->FUNCTIONAL data
f = resize_ts(f, resize="norm")
f = f - f.mean(axis=0)[np.newaxis, ...]
Ydiff = np.diff(f) #; print(Ydiff.shape)

#plot E, Ydiff and E-Ydiff
fig, axes = plt.subplots(2, 1, figsize=(10, 10))
pos_0 = axes[0].imshow((np.abs(E_paz-Ydiff))[:,:], interpolation='nearest', vmin=0, vmax=0.5); axes[0].set_title("abs(E_paz-Ydiff)")
pos_1 = axes[1].imshow(np.abs(E_lesioned-Ydiff), interpolation='nearest', vmin=0, vmax=0.5); axes[1].set_title("abs(E_lesioned-Ydiff)")
plt.tight_layout()
fig.savefig("all_Vs_few.png")
plt.show() 




