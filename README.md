# Development of early prediction model for pregnancy-associated hypertension with graph-based semi-supervised learning
Implementation of graph-based semi-supervised learning

### Requirements
This implementation is conducted by the following packages (to be installed independently)
  * pandas
  * numpy
  * sklearn
  * jupyter notebook: instructions of step-by-step process

### Run graph-based semi-supervised learning
Similar implementation of graph-based SSL can be found at scikit-learn (sklearn.semi_supervised.LabelPropagation)
```
Toy_Example_graph_based_SSL.ipynb
```
  This file explains step by step how to use graph-based SSL.
```  
Graph_based_SSL.py
test_main.py
```
  Functional version and toy example demonstration
  
### Note
  1. Implementation of variable selection also can be found at scikit-learn
    a. from sklearn.linear_model import LogisticRegression
    b. from sklearn.ensemble import ExtraTreesClassifier
    c. from sklearn.svm import SVC
    d. from sklearn.feature_selection import RFE
  
  2. To protect patient's private information, we can not share the data. (To reqeust data access, please contact authors: Seung Mi Lee. M.D., Ph.D.)
