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


"""
    -not0 : doesn't take into account self loops, not calculated tau0
    -snn--> not normalise the structural passed

    give as strucutral matrices the polinomial of the laplacian of strucutral matrice

    use powers of laplacian of structural matrix to fit the model (laplacian not normalised)
		-with and without self loops
		- NOT -not0 --> use L^0 = I
		- -s S, this way we are using L0(I), L1
		- -s S -snn L2 L3 ... (calculated laplancians before)

    NB laplacinae passed ARE NOT NORMLAISED
"""

#structural matrix
s = io.load_mat('raw/SC_avg56.mat')

data_path = "data/test15"
if not os.path.exists(data_path):
    os.makedirs(data_path)
L_path = "data/test15/power_L"
if not os.path.exists(L_path):
    os.makedirs(L_path)

#max power taht we want to test
n = 10 #max power will be n-1

calculations = 0

if calculations:

    #compute powers of the laplacian (of the strucutral matrix)
    L, degree = laplacian.compute_laplacian(s, selfloops="degree") #THEY ARE NOT NORMALISED

    for i in range(n): #NB 0,1 laplcansa will, not be used, only from 2 power ongoing
        L_power = np.linalg.matrix_power(L, i)
        io.export_mtx(L_power,f'{L_path}/L_{i}.mat') 

    for i in range(n):
        if i == 0: continue #makes no sense
        if i == 1: # -s S, this way we are using L0(I), L1
            #without tau 0
            crispy_gls_scalar.multitau_gls_estimation(tsfile = "raw/RS_1subj.mat",
            structural_files = f'raw/SC_avg56.mat', #f"raw/SC_avg56.mat",
            add_tau0 = False,
            sub = "1",
            odr = f"data/test15/{i}_powers_no_tau0")

            #with tau0
            crispy_gls_scalar.multitau_gls_estimation(tsfile = "raw/RS_1subj.mat",
            structural_files = f'raw/SC_avg56.mat', #calculate L^1
            add_tau0 = True, #calculate L^0 (I)
            sub = "1",
            odr = f"data/test15/{i}_powers_yes_tau0")

        L_to_use = [f"data/test15/power_L/L_" + str(j) + ".mat" for j in np.arange(2, i+1, 1)] # numbers: 2, 3, ..., i
        print(L_to_use)
        
        #without tau 0
        crispy_gls_scalar.multitau_gls_estimation(tsfile = "raw/RS_1subj.mat",
        structural_files = f'raw/SC_avg56.mat', ##calculate L^1
        structural_files_nonorm = L_to_use, #pass directly L^2, L^3, ...
        add_tau0 = False, #calculate L^0 (I)
        sub = "1",
        odr = f"data/test15/{i}_powers_no_tau0")

        #with tau 0
        crispy_gls_scalar.multitau_gls_estimation(tsfile = "raw/RS_1subj.mat",
        structural_files = f'raw/SC_avg56.mat', ##calculate L^1
        structural_files_nonorm = L_to_use, #pass directly L^2, L^3, ...
        add_tau0 = True, #calculate L^0 (I)
        sub = "1",
        odr = f"data/test15/{i}_powers_yes_tau0")

#list that contains lists of the taus for arch time we increase the power used

########################################
########################################
"""
    plot norm(E) Vs number of power used, find out if there is a plateau (agter a power the error doesn't diminish)
"""

from numpy.linalg import norm

Es_n0 = np.zeros(n)
Es_y0 = np.zeros(n)
x = np.arange(1, n, 1)

for i in np.arange(1, n, 1): #i don0t want 0 index
    print(f"extracting {i} error")
    Es_n0[i] = norm(io.load_txt(f"data/test15/{i}_powers_no_tau0/files/sub-1_ts-innov.tsv.gz"))
    Es_y0[i] = norm(io.load_txt(f"data/test15/{i}_powers_yes_tau0/files/sub-1_ts-innov.tsv.gz"))


fig, a = plt.subplots(1,1, dpi=300)
a.plot(x, Es_n0[1:], "o-", label="without self loops")
a.plot(x, Es_y0[1:], "o-", label="with self loops")
a.set_xlabel("Order of polynomial")
a.set_ylabel("norm(Error)")

plt.legend()

plt.savefig("data/test15/E_Vs_number_powers_used.png")

##########################################################
##########################################################

"""
    plot the taus for each order
"""
taus_y0 = [] #list pof lists of taus (one list foe each max power used)
taus_n0 = []

for i in np.arange(1, n, 1): #i don0t want 0 index
    if i == 1:
        with open(f"data/test15/{i}_powers_no_tau0/files/sub-1_tau_scalar.tsv", 'r', newline='', encoding='utf-8') as tsvfile:
            tsv = csv.reader(tsvfile, delimiter='\t')
            for row in tsv:
                # Assuming the file contains only one value, extract it from the first row and first column
                if len(row) > 0:
                    taus_n0.append(np.array([float(row[0])]))
                    break  # Exit the loop since the value is found
        #it has at least 2 taus sp can unse thi norma fucntion
        taus_y0.append(np.array(io.load_txt(f"data/test15/{i}_powers_yes_tau0/files/sub-1_tau_scalar.tsv")))

    else:
        taus_n0.append(np.array(io.load_txt(f"data/test15/{i}_powers_no_tau0/files/sub-1_tau_scalar.tsv")))
        taus_y0.append(np.array(io.load_txt(f"data/test15/{i}_powers_yes_tau0/files/sub-1_tau_scalar.tsv")))


#NB taus_y0 has always an elemtn in addition (the tua0)

fig, a = plt.subplots(1,1, dpi=300)

for i in np.arange(1, n, 1):
    i-=1 #ATTENTION indeed in the list i save withou L0
    print(f"power: ", {i+1})
    print("without self loop: ", taus_n0[i])
    print("with self loop: ", taus_y0[i])

    #print(len(taus_n0[i]), len(np.arange(1, len(taus_n0[i])+1, 1))) 
    #print(len(taus_y0[i]), len(np.arange(0, len(taus_y0[i]), 1))) 

    a.plot(np.arange(1, len(taus_n0[i])+1, 1), taus_n0[i], "o-", label=f"power:{i+1} without self loops", linewidth=2)
    a.plot(np.arange(0, len(taus_y0[i]), 1), taus_y0[i], "o-", label=F"power:{i+1} with self loops", linewidth=2)

a.set_xlabel("tau number")
a.set_ylabel("value of tau")

plt.legend(loc="best")

plt.savefig("data/test15/taus_differt_powers.png")


######################################################àà
#####################################################

"""
    plot the taus ONLY NO TAU0
"""
taus_n0 = []

for i in np.arange(1, n, 1): #i don0t want 0 index
    if i == 1:
        with open(f"data/test15/{i}_powers_no_tau0/files/sub-1_tau_scalar.tsv", 'r', newline='', encoding='utf-8') as tsvfile:
            tsv = csv.reader(tsvfile, delimiter='\t')
            for row in tsv:
                # Assuming the file contains only one value, extract it from the first row and first column
                if len(row) > 0:
                    taus_n0.append(np.array([float(row[0])]))
                    break  # Exit the loop since the value is found

    else:
        taus_n0.append(np.array(io.load_txt(f"data/test15/{i}_powers_no_tau0/files/sub-1_tau_scalar.tsv")))

#NB taus_y0 has always an elemtn in addition (the tua0)

fig, a = plt.subplots(1,1, dpi=300)

for i in np.arange(1, n, 1):
    i-=1 #ATTENTION indeed in the list i save withou L0

    a.plot(np.arange(1, len(taus_n0[i])+1, 1), taus_n0[i], "o-", label=f"power:{i+1} without self loops", linewidth=2)

a.set_xlabel("tau number")
a.set_ylabel("value of tau")

plt.legend(loc="best")


plt.savefig("data/test15/taus_N0_differt_powers.png")


######################################################àà
#####################################################

"""
    plot the taus ONLY YES TAU0
"""
taus_y0 = [] #list pof lists of taus (one list foe each max power used)

for i in np.arange(1, n, 1): #i don0t want 0 inde
    taus_y0.append(np.array(io.load_txt(f"data/test15/{i}_powers_yes_tau0/files/sub-1_tau_scalar.tsv")))

#NB taus_y0 has always an elemtn in addition (the tua0)

fig, a = plt.subplots(1,1, dpi=300)

for i in np.arange(1, n, 1):
    i-=1 #ATTENTION indeed in the list i save withou L0

    a.plot(np.arange(0, len(taus_y0[i]), 1), taus_y0[i], "o-", label=F"power:{i+1} with self loops", linewidth=2)

a.set_xlabel("tau number")
a.set_ylabel("value of tau")

plt.legend(loc="best")

plt.savefig("data/test15/taus_Y0_differt_powers.png")