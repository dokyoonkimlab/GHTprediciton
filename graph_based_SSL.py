
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.neighbors import kneighbors_graph


# arr : input matrix
# mu: trade-off parameter
# k: k-nn
# pos_idx: index for positive samples
# neg_idx: index for negative samples

# metics: canbe replaced other metric

def graph_based_SSL(arr, mu, gamma, k, pos_idx, neg_idx):    
    N_nodes = len(arr)
    Identity = np.eye(N_nodes)
    Dist = kneighbors_graph(arr, k, mode='connectivity', metric='euclidean') 
    Dist = Dist.toarray()
    upper_dist = np.triu(Dist)
    lower_dist = np.tril(Dist)
    Incidence = upper_dist + lower_dist + upper_dist.transpose()+lower_dist.transpose()
    
    Incidence = np.where(Incidence >= 1, 1, 0)
    Dist = metrics.pairwise.euclidean_distances(arr) * Incidence
    
    Weight = np.exp(-(Dist ** 2 / gamma ** 2)) # gamma == 1
    
    Degree = np.diag(sum(Weight), 1)

    Laplacian = Degree - Weight
            
    Laplacian = Identity + mu * Laplacian
    Inverse_Laplacian = np.linalg.inv(Laplacian)
   
    y = np.zeros(N_nodes) # unlabeled set to 0
    y[pos_idx] = 1
    y[neg_idx] = -1 
    
    f = Inverse_Laplacian@y
    return f