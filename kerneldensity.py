# exercise 11.4.1
import numpy as np
from matplotlib.pyplot import (figure, bar, title, xticks, yticks, cm,
                               subplot, show, ylabel, scatter)
from scipy.io import loadmat
from toolbox_02450 import gausKernelDensity
from sklearn.neighbors import NearestNeighbors
from dataextraction import X_standard

# load data
X = X_standard
# y = X[:,9]
# X = X[:,:9]
N, M = np.shape(X)
n = 60

### Gausian Kernel density estimator
# cross-validate kernel width by leave-one-out-cross-validation
# (efficient implementation in gausKernelDensity function)
# evaluate for range of kernel widths
widths = X.var(axis=0).max() * (2.0**np.arange(-10,3))
logP = np.zeros(np.size(widths))
for i,w in enumerate(widths):
   print('Fold {:2d}, w={:f}'.format(i,w))
   density, log_density = gausKernelDensity(X,w)
   logP[i] = log_density.sum()
   
val = logP.max()
ind = logP.argmax()

width=widths[ind]
print('Optimal estimated width is: {0}'.format(width))

# evaluate density for estimated width
density, log_density = gausKernelDensity(X,width)

# Sort the densities
i = (density.argsort(axis=0)).ravel()
outliersgkd = i[:28]
print("outliers gkd:", outliersgkd)

density = density[i].reshape(-1,)

# Plot density estimate of outlier score
subplot(2,2,1)
b1 = bar(range(n),density[:n])
for index in range(28):
   b1[index].set_color('r')
ylabel("Outlier score")
title('Kernel density estimate')

### K-neighbors density estimator
# Neighbor to use:
K = 5

# Find the k nearest neighbors
knn = NearestNeighbors(n_neighbors=K).fit(X)
D, i = knn.kneighbors(X)

density = 1./(D.sum(axis=1)/K)

# Sort the scores
i = density.argsort()

outliersknn = i[:8]
print("outliers knn:", outliersknn)

density = density[i]
subplot(2,2,2)
b2 = bar(range(n),density[:n])
for index in range(8):
   b2[index].set_color('r')
ylabel("Outlier score")
title('KNN density: Outlier score')


### K-nearest neigbor average relative density
# Compute the average relative density

knn = NearestNeighbors(n_neighbors=K).fit(X)
D, i = knn.kneighbors(X)
density = 1./(D.sum(axis=1)/K)
avg_rel_density = density/(density[i[:,1:]].sum(axis=1)/K)

# Sort the avg.rel.densities
i_avg_rel = avg_rel_density.argsort()
avg_rel_density = avg_rel_density[i_avg_rel]

outliersard = i_avg_rel[:6]
print("outliers ard:", outliersard)


subplot(2,2,3)
b3 = bar(range(n),avg_rel_density[:n])
for index in range(6):
   b3[index].set_color('r')
title('KNN average relative density: Outlier score')
ylabel("Outlier score")

### Distance to 5'th nearest neighbor outlier score
K = 5

# Find the k nearest neighbors
knn = NearestNeighbors(n_neighbors=K).fit(X)
D, i = knn.kneighbors(X)

# Outlier score
score = D[:,K-1]
# Sort the scores
i = score.argsort()
score = score[i[::-1]]

outliers5 = i[:8]
print("outliers 5thnn:", outliers5)


subplot(2,2,4)
b4=bar(range(n),score[:n])
for index in range(8):
   b4[index].set_color('r')
title('5th neighbor distance: Outlier score')
ylabel("Outlier score")
show()

a = set(outliersgkd)
b = set(outliersknn)
c = set(outliersard)
d = set(outliers5)

print(a & b & c)
print('ad', a & d)
print('cd', c & d)
print('bd', b & d)


# PCA stuff
from pcavariance import (N, M, Y, U, S, V)
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

densities = [outliersgkd, outliersknn, outliersard, outliers5]
titles = ['gkd', 'knn', 'ard', '5th nearest neighbor']
for i, d in enumerate(densities):
   print(i)
   subplot(2, 2, i+1)
   title(titles[i])
   for index in range(len(Z)):
      if index in d:
         scatter(Z[index][0], Z[index][1], c='r', s=8)
      else:
         scatter(Z[index][0], Z[index][1], c='g', s=8)
show()
