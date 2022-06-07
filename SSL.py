# -*- coding: utf-8 -*-
"""
Created on Tue Jun  7 17:58:05 2022

@author: namyh123
"""


import numpy as np
from sklearn import metrics
from sklearn.neighbors import kneighbors_graph


# arr : input matrix
# mu: trade-off parameter
# k: k-nn
# pos_idx: index for positive samples
# neg_idx: index for negative samples

# metics: canbe replaced other metric

def models(arr, mu, gamma, k, pos_idx, neg_idx):    
    N_nodes = len(arr)
    
    Identity = np.eye(N_nodes)
    
    Adj_k = kneighbors_graph(arr, 20, mode='connectivity', metric='euclidean')  # X input matrix,  Connectivity: 20-Nearest neighbors
    Adj_k = Adj_k.toarray() # convert adjacency list to adjancecy matrix

    # Convert symmetric adj matrix: Function (kneighbors_graph) cannot guarantee symmetric matrix
    Adj_k = np.triu(Adj_k) + np.tril(Adj_k) + np.triu(Adj_k).transpose()+np.tril(Adj_k).transpose()
    Adj_k = np.where(Adj_k >= 1, 1, 0) # symmetric connectivty for k-NN
    
    Dist = metrics.pairwise.euclidean_distances(arr, arr)
    Weight = np.exp(-Dist / gamma ** 2) 
    
    Weight_k = Weight * Adj_k + Identity
    
    Degree = np.diag(np.sum(Weight_k, 1))

    Laplacian = Degree - Weight_k
            
    Laplacian = Identity + mu * Laplacian
    Inverse_Laplacian = np.linalg.inv(Laplacian)
   
    y = np.zeros(N_nodes) # unlabeled set to 0
    y[pos_idx] = 1
    y[neg_idx] = -1 
    
    f = Inverse_Laplacian@y
    return f