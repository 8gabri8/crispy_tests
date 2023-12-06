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

"""
    create synthethic ts where tau and strucral matrices are know
    test if the script that uses laplacina as strucutral file is able to predicit the cirrect taus
"""

def generate_and_compare_taus(n_roi=6, n_tps=500, perc_spike=0.2, yes_noise=False, NOISE=1, n=3, seed = 42):

    #set a seed so is the same between laplacins and eigendevompsition
    np.random.seed(seed)

    path = "data/test21"
    if not os.path.exists(f"{path}"):
        os.mkdir(f"{path}")
    L_path = "data/test21/power_L"
    if not os.path.exists(L_path):
        os.makedirs(L_path)

    ############################################
    ### CREATE SC MATRIX
    ############################################

    mtx = np.random.rand(n_roi, n_roi)
    mtx = (mtx + mtx.T) / 2
    mtx = mtx - mtx.min()
    mtx[np.diag_indices(mtx.shape[0])] = 0

    io.export_mtx(mtx, f"{path}/SC.mat")

    ############################################
    ### CREATE TS
    ############################################

    ts = np.zeros((n_roi, n_tps)) #ts empty

    # will contain where the spike are in the tsù
    ## Fix spike to be before 2/3 of ts

    #one spike for each row
    # ts_spike = []
    # for i in range(n_roi):
    #     temp = np.zeros(n_tps)
    #     k = np.random.randint(0, int((2/3) * n_tps))
    #     temp[k] = 1
    #     ts_spike.append(temp)
    # ts_spike = np.array(ts_spike, dtype=int) 

    #more spikes per rows
    ts_spike = np.zeros((n_roi, n_tps))
    for i in range(int(n_tps * perc_spike)):
        row = np.random.randint(0, n_roi)
        column = np.random.randint(0, int((2/3) * n_tps))
        ts_spike[row, column] = 1

    NOISE = 1
    if yes_noise:
        NOISE = 0.01
    ts_random = np.random.randn(n_roi, n_tps) * NOISE

    ############################################
    ### PLOT TS AND SC
    ############################################

    plot_greyplot(ts_spike,f"{path}/ts_spike.png")
    plot_greyplot(mtx,f"{path}/mtx_sc_synth.png")

    ############################################
    ### CONSTRUCT LAPLACIANS
    ############################################

    #n = 6 #max power is n-1 (the first matricxi is L^0=I)
        #ex. n=3 L0, L1, L2, L3
    n +=1

    L, degree = laplacian.compute_laplacian(mtx, selfloops="degree")
    L = laplacian.normalisation(L, degree, norm="rwo")

    io.export_mtx(np.identity(n_roi), f'{L_path}/L_0.mat') #L^0 =  Identity

    for i in np.arange(1,n): 
        L_power = np.linalg.matrix_power(L, i)
        #L_power = laplacian.normalisation(L_power, degree, norm="rwo")
        io.export_mtx(L_power,f'{L_path}/L_{i}.mat') 


    ############################################
    ### CHOOSE TAUS
    ############################################

    # Create tau_vec as an array of random positive values between 0.02 and 0.1
    #ATTENTION IF TAU IS TOO BIG THERE IS A NON CONVERGENCE
    taus = np.random.uniform(low=0.02, high=0.1, size=(n,)) #we want also tau0

    #try tau in ascdending order, so tau0 is the smallest
    #taus.sort()

    #try negative taus
    #taus = np.random.uniform(low=-0.1, high=0.1, size=(n,)) #we want also tau0

    io.export_mtx(taus, f'{path}/taus_GT.mat') 

    ############################################
    ### CONSTRUCT SYNTHEIC SIGNAL Yn
    ############################################

    Y = np.zeros((n_roi, n_tps))
    Y[:,0] = np.zeros(n_roi)

    #Yn = [ (1 - tau0) I - tau1 L1 - tau2 L2 - ...] Yn-1 + En
    for i in np.arange(1, n_tps): # foer each timeporint from the second one
        coeff = (1 - taus[0]) * np.identity(n_roi) #can also use L_0.mat

        for j in np.arange(1, n): #rest of the summation    
            coeff -= taus[j] * io.load_mat(f'{L_path}/L_{j}.mat')

        Y[:,i] = coeff @ Y[:, i-1]

        Y[:,i] += ts_spike[:,i] #add spike in this moment

        if yes_noise:
            Y[:,i] += ts_random[:, i]


    io.export_mtx(Y, f"{path}/Y_synth.mat")

    ############################################
    ### FROM THE SYNTH SINGNAL TRY TO RECONSTRUCT TAUS
    ############################################

    path_struct = [f"{L_path}/L_" + str(i) + ".mat" for i in np.arange(2, n)]

    crispy_gls_scalar.multitau_gls_estimation(tsfile = f"{path}/Y_synth.mat",
                    add_tau0 = True, #--> L^0 = Identity
                    structural_files = f"{path}/SC.mat", #strucral matric --> L^1
                    structural_files_nonorm = path_struct, #Laplacinasa are already normalised
                    sub = "1",
                    odr = f"{path}/diffusion_model")


    ############################################
    ### COMPARE TAUS
    ############################################

    taus_GT = io.load_mat(f"{path}/taus_GT.mat")
    taus_predicted = np.array(io.load_txt(f"{path}/diffusion_model/files/sub-1_tau_scalar.tsv"))

    return taus, taus_predicted

gt = []
pred = []
n=15 #max numbee of pawer of laplacina
n_roi=60
n_tps=2000
n_seeds = 10

#FOR A SINGLE ROUND

# #NB we must have at least one matrix
# for i in range(1,n):
#     a, b = generate_and_compare_taus(n_roi=60, n_tps=2000,n=i,seed=13)
#     gt.append(a)
#     pred.append(b)

# fig, a = plt.subplots(n-1,2,dpi=300,figsize=(5,10))
# for i in range(n-1):
#     a[i,0].plot(gt[i])
#     a[i,0].set_title(f"{i+1} Taus")
#     a[i,1].plot(pred[i])
# plt.tight_layout()
# plt.savefig(f"data/test21/more_L.png")


# #PLOT RNAGE DEPENDIG ON HOW MANY TAUS ARE USED

# fig, a = plt.subplots(1,1,dpi=300,figsize=(5,5))
# std = []
# for i in range(n-1):
#     std.append(np.abs(np.max(pred[i]) - np.min(pred[i])))
# a.plot(std)
# a.set_title(f"Nodes: {n_roi} Time Points: {n_tps}")
# a.set_xlabel("Taus Used")
# a.set_ylabel("Range")
# plt.tight_layout()
# plt.savefig(f"data/test21/ranges.png")

def more_ranges(seed, n_roi, n_tps, n):
    gt = []
    pred = []

    for i in range(1,n):
        a, b = generate_and_compare_taus(n_roi = n_roi, n_tps = n_tps,n=i,seed=seed)
        gt.append(a)
        pred.append(b)

    ranges= []
    for i in range(n-1):
        ranges.append(np.abs(np.max(pred[i]) - np.min(pred[i])))

    return ranges

ranges = []
for i in (range(n_seeds))*7:
    print(i)
    ranges.append(more_ranges(seed=i, n_roi=n_roi, n_tps=n_tps, n=n))

r = np.array(ranges)
print(r.shape)
print(r)
plt.clf()
for o in r:
    plt.plot(o)
plt.savefig("data/test21/all_ranges_different_seed.png")

r = np.mean(r, axis = 0)
print(f"\n{r}")
plt.clf()
plt.plot(r)
plt.savefig("data/test21/mean_ranges_different_seed.png")