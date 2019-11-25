from dataextraction import X, X_standard, attributeNames
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pcavariance import (X, N, M, Y, U, S, V)

import matplotlib.pyplot as plt
from toolbox_02450 import clusterplot
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram

# Perform hierarchical/agglomerative clustering on data matrix
## single, complete, average, weighted, centroid, median, ward
Method = 'ward'
Metric = 'euclidean'
X = X_standard
y = X[:,9] #make y the chd column
X = X[:,:9] #get rid of chd column

# PCA stuff so that clustering is graphed on PCA 2d projection
Vt = V.T
Z = Y @ V
z0 = []
z1 = []
counter0 = 0
counter1 = 0
for r in range(len(Z)):
    if Y[r][9] == 1:
        z1 += [0]
        z1[counter1] = Z[r]
        counter1 += 1
    else:
        z0 += [0]
        z0[counter0] = Z[r]
        counter0 += 1
z0 = np.array(z0)
z1 = np.array(z1)

link = linkage(X, method=Method, metric=Metric)

# Compute clusters by thresholding the dendrogram
Maxclust = 2
cls = fcluster(link, criterion='maxclust', t=Maxclust)

# Find out accuracy of clusters
a = cls.reshape(cls.shape[0],1)
truepos = 0
trueneg = 0
falsepos = 0
falseneg = 0
for index, item in enumerate(a):
    if item-1 == 0 and y[index] == 0:
        trueneg += 1
    elif item-1 == 1 and y[index] == 1:
        truepos += 1
    elif item-1 == 0 and y[index] == 1:
        falseneg += 1
    elif item-1 == 1 and y[index] == 0:
        falsepos += 1
    else:
        print("something weird", index, item)
print("Method:",Method)
print("truepos:", truepos)
print("trueneg:", trueneg)
print("falsepos:", falsepos)
print("falseneg:", falseneg)
print("Percent right:", (truepos + trueneg)/len(a) * 100)

# Plot clusters
plt.figure(1)
clusterplot(Z[:,0:2], a, y=y)

# Display dendrogram
max_display_levels=6
plt.figure(2,figsize=(10,4))
dendrogram(link, truncate_mode='level', p=max_display_levels)

plt.show()