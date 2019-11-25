from dataextraction import X

import matplotlib.pyplot as plt
from toolbox_02450 import clusterplot
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram

# Perform hierarchical/agglomerative clustering on data matrix
Method = 'single'
Metric = 'euclidean'

Z = linkage(X, method=Method, metric=Metric)

# Compute and display clusters by thresholding the dendrogram
Maxclust = 2
cls = fcluster(Z, criterion='maxclust', t=Maxclust)
plt.figure(1)
clusterplot(X, cls.reshape(cls.shape[0],1), y=X[:,9])

# Display dendrogram
max_display_levels=6
plt.figure(2,figsize=(10,4))
dendrogram(Z, truncate_mode='level', p=max_display_levels)

plt.show()

print('Ran Exercise 10.2.1')