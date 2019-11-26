from dataextraction import X, X_standard, attributeNames 
import numpy as np
import matplotlib.pyplot as plt
from pcavariance import (N, M, Y, U, S, V)
from toolbox_02450 import clusterplot
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram

# Perform hierarchical/agglomerative clustering on data matrix
## single, complete, average, weighted, centroid, median, ward

Method = 'ward' 
#possibilities:
#single
#average
#weighted
#centroid
#median
#ward

Metric = 'euclidean'
#possibilities:
#https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.pdist.html#scipy.spatial.distance.pdist


X = X_standard
y = X[:,9] #make y the chd column
X = X[:,:9] #get rid of chd column
# y = X[:,4]
# X = X[:,[0,1,2,3,5,6,7,8,9]]


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

# link = linkage(X, method=Method, metric=Metric)

# # Compute clusters by thresholding the dendrogram
# Maxclust = 2
# cls = fcluster(link, criterion='maxclust', t=Maxclust)

# # Find out accuracy of clusters
# a = cls.reshape(cls.shape[0],1)
# truepos = 0
# trueneg = 0
# falsepos = 0
# falseneg = 0
# for index, item in enumerate(a):
#     if item-1 == 0 and y[index] == 0:
#         trueneg += 1
#     elif item-1 == 1 and y[index] == 1:
#         truepos += 1
#     elif item-1 == 0 and y[index] == 1:
#         falseneg += 1
#     elif item-1 == 1 and y[index] == 0:
#         falsepos += 1
#     else:
#         print("something weird", index, item)
# print("Method:",Method)
# print("truepos:", truepos)
# print("trueneg:", trueneg)
# print("falsepos:", falsepos)
# print("falseneg:", falseneg)
# print("Percent right:", (truepos + trueneg)/len(a) * 100)

# # Plot clusters
# plt.figure(1, figsize=(10,8))
# clusterplot(Z[:,0:2], a, y=y)
# plt.xlabel("PC1")
# plt.ylabel("PC2")
# plt.title("Hierarchical clustering")

# # Display dendrogram
# max_display_levels=6
# plt.figure(2,figsize=(11,4))
# dendrogram(link, truncate_mode='level', p=max_display_levels)

# plt.show()

#%% testing all posibilites 
def testAll():
    Maxclust = 2
    methods = ["single","average","weighted","centroid","median","ward"]
    metrics = ["euclidean","minkowski","cityblock","seuclidean","cosine","correlation","hamming","jaccard","chebyshev","canberra","braycurtis","mahalanobis","yule","matching","dice","kulsinski","regoerstanimoto","russellrao","sokalsneath","wminkowski"]

    for method in methods:
        for metric in metrics:
            

            # Compute clusters by thresholding the dendrogram
            try:
                
                link = linkage(X, method=method, metric=metric)
                cls = fcluster(link, criterion='maxclust', t=Maxclust)
                print("Method used is",method)
                print("Metric used is",metric,"\n")
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
                print("\n--------------------------------------------")
            except:
                pass
            
#%% new y is famhist to see if we can improve result
X = X_standard
y = X[:,4]
X = X[:,[0,1,2,3,5,6,7,8,9]]

link = linkage(X, method="average", metric="matching") #metric=hamming

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
        print("something weird", index, item, y[index])
print("Method:",Method)
print("truepos:", truepos)
print("trueneg:", trueneg)
print("falsepos:", falsepos)
print("falseneg:", falseneg)
print("Percent right:", (truepos + trueneg)/len(a) * 100)

# Plot clusters
plt.figure(1, figsize=(10,8))
clusterplot(Z[:,0:2], a, y=y)
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("Hierarchical clustering for predicting famhist")

# Display dendrogram
max_display_levels=6
plt.figure(2,figsize=(11,4))
dendrogram(link, truncate_mode='level', p=max_display_levels)

plt.show()