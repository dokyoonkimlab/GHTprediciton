
import numpy as np
import pandas as pd
from sklearn import metrics
import SSL

# Load Toy dataset
from sklearn import datasets
X, y = datasets.load_breast_cancer(return_X_y = True)

# Scaling
from sklearn.preprocessing import StandardScaler
Scaler = StandardScaler()
X = StandardScaler().fit_transform(X) # centering

####################### Generating Unlabeleld Set #####################
N_samples = len(X)

# class index
idx_pos = np.where(y==1)[0]
idx_neg = np.where(y==0)[0]

ground_truth = np.zeros(N_samples)
ground_truth[idx_pos] = 1
ground_truth[idx_neg] = -1

Masking_ratio = 0.6  # making: 60% labeled to unlabeled data

# Randomly sampling unlabeled idx
import random
shuffle = np.array(random.sample(range(N_samples), N_samples))
N_unlabeled = round(N_samples * Masking_ratio) # number of unlabeled samples

unlabeled_idx = shuffle[0:N_unlabeled]
print(unlabeled_idx)

pos_idx_init = np.setdiff1d(idx_pos, unlabeled_idx)
neg_idx_init = np.setdiff1d(idx_neg, unlabeled_idx)
######################################################################


##### Run SSL
f = SSL.models(arr=X, mu=0.5, gamma=1, k=20, pos_idx=pos_idx_init, neg_idx=neg_idx_init)


##### AUC calculation
f_unlabeled = pd.DataFrame( f[unlabeled_idx], columns={'predicted'} )
y_unlabeled = pd.DataFrame( ground_truth[unlabeled_idx], columns={'actual'})

fpr, tpr, th = metrics.roc_curve(y_unlabeled, f_unlabeled, pos_label=1)
AUC = metrics.auc(fpr, tpr)

print("AUC:: ", AUC)