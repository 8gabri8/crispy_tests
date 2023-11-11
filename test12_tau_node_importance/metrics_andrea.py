from nigsp import io, viz
import numpy as np
import os
from matplotlib import pyplot as plt
import pandas as pd
import csv
from nigsp.operations.timeseries import resize_ts
from scipy.interpolate import make_interp_spline
import networkx as nx


def order_taus(index):
    taus_lesion = []
    taus_paz = []
    taus_1_paz_lesion = []
    taus_2_paz_lesion = []

    for label in index:
        with open(f"lesion/lesion_{label}/files/sub-1_tau_scalar.tsv", 'r', newline='', encoding='utf-8') as tsvfile:
            tsv = csv.reader(tsvfile, delimiter='\t')
            for row in tsv:
                # Assuming the file contains only one value, extract it from the first row and first column
                if len(row) > 0:
                    taus_lesion.append(float(row[0]))
                    break  # Exit the loop since the value is found

        with open(f"paz/paz_{label}/files/sub-1_tau_scalar.tsv", 'r', newline='', encoding='utf-8') as tsvfile:
            tsv = csv.reader(tsvfile, delimiter='\t')
            for row in tsv:
                # Assuming the file contains only one value, extract it from the first row and first column
                if len(row) > 0:
                    taus_paz.append(float(row[0]))
                    break  # Exit the loop since the value is found
        
        with open(f"paz-lesion/paz-lesion_{label}/files/sub-1_tau_scalar.tsv", 'r', newline='', encoding='utf-8') as tsvfile:
            tsv = csv.reader(tsvfile, delimiter='\t')
            k=1 #index of the tau to read
            for row in tsv:
                if len(row) > 0:
                    if k==1:
                        taus_1_paz_lesion.append(float(row[0]))
                    elif k==2:
                        taus_2_paz_lesion.append(float(row[0]))
                k+=1

    return taus_lesion, taus_paz, taus_1_paz_lesion, taus_2_paz_lesion


# structral  matrix
s = io.load_mat("raw/SC_avg56.mat") 
S = nx.from_numpy_array(s)


#########################
### BETWEENNESS CENTRALITY
########################
"""
    For every pair of vertices in a connected graph, 
    there exists at least one shortest path between 
    the vertices such that either the number of edges that 
    the path passes through (for unweighted graphs) or the sum 
    of the weights of the edges (for weighted graphs) is minimized. 
    The betweenness centrality for each vertex is the number 
    of these shortest paths that pass through the vertex.
"""

data_path = "data/test12/metrics_andrea"
if not os.path.exists(data_path):
    os.makedirs(data_path)

bet_centrality = nx.betweenness_centrality(S) #want a nx-graph return a dictionary, node : value of centrality
bet_centrality = list(bet_centrality.values())
bet_centrality = np.array(bet_centrality)
print(bet_centrality.shape)
index = np.argsort(bet_centrality)

#they are odered in increaisng order by this specific metric (i.e. bet_centrlity)
taus_lesion, taus_paz, taus_1_paz_lesion, taus_2_paz_lesion = order_taus(index)

                   
fig, a = plt.subplots(1, 2, figsize=(25,10))#, gridspec_kw={'width_ratios': [0.99,0.01]})
cn = np.arange(len(index))
x_ticks = []
for i in index:
    x_ticks.append(f"{i}_{bet_centrality[i]:.2f}")
a[0].set_xticks(cn, index, rotation='vertical') #the xticks are the "name"/Id of the node
a[1].set_xticks(cn, index, rotation='vertical') #the xticks are the "name"/Id of the node

a[0].plot(cn, taus_lesion, label=f"lesion, CORR = {np.round(np.corrcoef(taus_lesion, bet_centrality[index])[1,0],2)}")
a[0].plot(cn, taus_paz, label=f"connectome, CORR = {np.round(np.corrcoef(taus_paz, bet_centrality[index])[1,0],2)}")
a[1].plot(cn, taus_1_paz_lesion, label=f"taus_1_lesion, CORR = {np.round(np.corrcoef(taus_1_paz_lesion, bet_centrality[index])[1,0],2)}")
a[1].plot(cn, taus_2_paz_lesion, label=f"taus_2_connectome, CORR = {np.round(np.corrcoef(taus_2_paz_lesion, bet_centrality[index])[1,0],2)}")

a[1].plot(np.arange(len(index)), np.interp(bet_centrality[index], (bet_centrality[index].min(), bet_centrality[index].max()), (0, 0.13)), label = "bet_centrality per node (scaled)")
a[0].plot(np.arange(len(index)), np.interp(bet_centrality[index], (bet_centrality[index].min(), bet_centrality[index].max()), (0, 0.13)), label = "bet_centrality per node (scaled)")

a[0].set_xlabel('nodes ordered by bet_centrality')
a[0].set_ylabel('tau')
a[0].legend(loc="best", fontsize=15)
a[0].title.set_text("lesion connectome separated")
a[1].title.set_text("lesion connectome together")
a[1].set_xlabel('nodes ordered by bet_centrality')
a[1].set_ylabel('tau')
a[1].legend(loc="best", fontsize=15)
fig.savefig("data/test12/metrics_andrea/taus_Vs_bet_centrality.png")
plt.show()


#########################
### CLOSENESS CENTRALITY
########################
"""
    reciprocal of the sum of the length of the shortest 
    paths between the node and all other nodes in the graph. 
    Thus, the more central a node is, the closer it is to all other nodes.
"""

clos_centrality = nx.closeness_centrality(S) #want a nx-graph return a dictionary, node : value of centrality
clos_centrality = list(clos_centrality.values())
clos_centrality = np.array(clos_centrality)
print(clos_centrality.shape)
index = np.argsort(clos_centrality)

#they are odered in increaisng order by this specific metric (i.e. bet_centrlity)
taus_lesion, taus_paz, taus_1_paz_lesion, taus_2_paz_lesion = order_taus(index)

                   
fig, a = plt.subplots(1, 2, figsize=(25,10))#, gridspec_kw={'width_ratios': [0.99,0.01]})
cn = np.arange(len(index))
x_ticks = []
for i in index:
    x_ticks.append(f"{i}_{clos_centrality[i]:.2f}")
a[0].set_xticks(cn, index, rotation='vertical') #the xticks are the "name"/Id of the node
a[1].set_xticks(cn, index, rotation='vertical') #the xticks are the "name"/Id of the node

a[0].plot(cn, taus_lesion, label=f"lesion, CORR = {np.round(np.corrcoef(taus_lesion, clos_centrality[index])[1,0],2)}")
a[0].plot(cn, taus_paz, label=f"connectome, CORR = {np.round(np.corrcoef(taus_paz, clos_centrality[index])[1,0],2)}")
a[1].plot(cn, taus_1_paz_lesion, label=f"taus_1_lesion, CORR = {np.round(np.corrcoef(taus_1_paz_lesion, clos_centrality[index])[1,0],2)}")
a[1].plot(cn, taus_2_paz_lesion, label=f"taus_2_connectome, CORR = {np.round(np.corrcoef(taus_2_paz_lesion, clos_centrality[index])[1,0],2)}")

a[1].plot(np.arange(len(index)), np.interp(clos_centrality[index], (clos_centrality[index].min(), clos_centrality[index].max()), (0, 0.13)), label = "clos_centrality per node (scaled)")
a[0].plot(np.arange(len(index)), np.interp(clos_centrality[index], (clos_centrality[index].min(), clos_centrality[index].max()), (0, 0.13)), label = "clos_centrality per node (scaled)")

a[0].set_xlabel('nodes ordered by clos_centrality')
a[0].set_ylabel('tau')
a[0].legend(loc="best", fontsize=15)
a[0].title.set_text("lesion connectome separated")
a[1].title.set_text("lesion connectome together")
a[1].set_xlabel('nodes ordered by clos_centrality')
a[1].set_ylabel('tau')
a[1].legend(loc="best", fontsize=15)
fig.savefig("data/test12/metrics_andrea/taus_Vs_clos_centrality.png")
plt.show()

#########################
### LOCAL CLUSTERING COEFF
########################
"""
    The local clustering coefficient C_{i} for a vertex 
    v_{i} is then given by a proportion of the number of 
    links between the vertices within its neighbourhood divided 
    by the number of links that could possibly exist between them
"""

clustering_coeff = nx.clustering(S) #want a nx-graph return a dictionary, node : value of centrality
clustering_coeff = list(clustering_coeff.values())
clustering_coeff = np.array(clustering_coeff)
print(clustering_coeff.shape)
index = np.argsort(clustering_coeff)

#they are odered in increaisng order by this specific metric (i.e. bet_centrlity)
taus_lesion, taus_paz, taus_1_paz_lesion, taus_2_paz_lesion = order_taus(index)

                   
fig, a = plt.subplots(1, 2, figsize=(25,10))#, gridspec_kw={'width_ratios': [0.99,0.01]})
cn = np.arange(len(index))
x_ticks = []
for i in index:
    x_ticks.append(f"{i}_{clustering_coeff[i]:.2f}")
a[0].set_xticks(cn, index, rotation='vertical') #the xticks are the "name"/Id of the node
a[1].set_xticks(cn, index, rotation='vertical') #the xticks are the "name"/Id of the node

a[0].plot(cn, taus_lesion, label=f"lesion, CORR = {np.round(np.corrcoef(taus_lesion, clustering_coeff[index])[1,0],2)}")
a[0].plot(cn, taus_paz, label=f"connectome, CORR = {np.round(np.corrcoef(taus_paz, clustering_coeff[index])[1,0],2)}")
a[1].plot(cn, taus_1_paz_lesion, label=f"taus_1_lesion, CORR = {np.round(np.corrcoef(taus_1_paz_lesion, clustering_coeff[index])[1,0],2)}")
a[1].plot(cn, taus_2_paz_lesion, label=f"taus_2_connectome, CORR = {np.round(np.corrcoef(taus_2_paz_lesion, clustering_coeff[index])[1,0],2)}")

a[1].plot(np.arange(len(index)), np.interp(clustering_coeff[index], (clustering_coeff[index].min(), clustering_coeff[index].max()), (0, 0.13)), label = "clustering_coeff per node (scaled)")
a[0].plot(np.arange(len(index)), np.interp(clustering_coeff[index], (clustering_coeff[index].min(), clustering_coeff[index].max()), (0, 0.13)), label = "clustering_coeff per node (scaled)")

a[0].set_xlabel('nodes ordered by clustering_coeff')
a[0].set_ylabel('tau')
a[0].legend(loc="best", fontsize=15)
a[0].title.set_text("lesion connectome separated")
a[1].title.set_text("lesion connectome together")
a[1].set_xlabel('nodes ordered by clustering_coeff')
a[1].set_ylabel('tau')
a[1].legend(loc="best", fontsize=15)
fig.savefig("data/test12/metrics_andrea/taus_Vs_clustering_coeff.png")
plt.show()

#########################
### PAGE RANK
########################
"""
    PageRank is a link analysis algorithm and it assigns a numerical
    weighting to each element of a hyperlinked set of documents, 
    such as the World Wide Web, with the purpose of "measuring" 
    its relative importance within the set.

"""

pagerank = nx.pagerank(S) #want a nx-graph return a dictionary, node : value of centrality
pagerank = list(pagerank.values())
pagerank = np.array(pagerank)
print(pagerank.shape)
index = np.argsort(pagerank)

#they are odered in increaisng order by this specific metric (i.e. bet_centrlity)
taus_lesion, taus_paz, taus_1_paz_lesion, taus_2_paz_lesion = order_taus(index)

                   
fig, a = plt.subplots(1, 2, figsize=(25,10))#, gridspec_kw={'width_ratios': [0.99,0.01]})
cn = np.arange(len(index))
x_ticks = []
for i in index:
    x_ticks.append(f"{i}_{pagerank[i]:.2f}")
a[0].set_xticks(cn, index, rotation='vertical') #the xticks are the "name"/Id of the node
a[1].set_xticks(cn, index, rotation='vertical') #the xticks are the "name"/Id of the node

a[0].plot(cn, taus_lesion, label=f"lesion, CORR = {np.round(np.corrcoef(taus_lesion, pagerank[index])[1,0],2)}")
a[0].plot(cn, taus_paz, label=f"connectome, CORR = {np.round(np.corrcoef(taus_paz, pagerank[index])[1,0],2)}")
a[1].plot(cn, taus_1_paz_lesion, label=f"taus_1_lesion, CORR = {np.round(np.corrcoef(taus_1_paz_lesion, pagerank[index])[1,0],2)}")
a[1].plot(cn, taus_2_paz_lesion, label=f"taus_2_connectome, CORR = {np.round(np.corrcoef(taus_2_paz_lesion, pagerank[index])[1,0],2)}")

a[1].plot(np.arange(len(index)), np.interp(pagerank[index], (pagerank[index].min(), pagerank[index].max()), (0, 0.13)), label = "pagerank per node (scaled)")
a[0].plot(np.arange(len(index)), np.interp(pagerank[index], (pagerank[index].min(), pagerank[index].max()), (0, 0.13)), label = "pagerank per node (scaled)")

a[0].set_xlabel('nodes ordered by pagerank')
a[0].set_ylabel('tau')
a[0].legend(loc="best", fontsize=15)
a[0].title.set_text("lesion connectome separated")
a[1].title.set_text("lesion connectome together")
a[1].set_xlabel('nodes ordered by pagerank')
a[1].set_ylabel('tau')
a[1].legend(loc="best", fontsize=15)
fig.savefig("data/test12/metrics_andrea/taus_Vs_pagerank.png")
plt.show()