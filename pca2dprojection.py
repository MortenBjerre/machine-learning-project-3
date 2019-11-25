import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pcavariance import (X, N, M, Y, U, S, V)
from dataextraction import attributeNames

Vt = V.T

# Project the centered data onto principal component space
Z = Y @ V

# Indices of the principal components to be plotted
i = 0
j = 1

# Plot PCA of the data
#f = plt.figure()
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

plt.scatter(z0[:,i], z0[:,j], c='g', alpha=0.5, s=10, label = "chd=0")
plt.scatter(z1[:,i], z1[:,j], c='r', alpha=0.5, s=10, label = "chd=1")
plt.title('heart disease data: PCA')
plt.legend()
plt.xlabel('PC{0}'.format(i+1))
plt.ylabel('PC{0}'.format(j+1))
plt.show()

# ax = plt.axes(projection='3d')
# ax.scatter3D(Z[:,0],Z[:,1],Z[:,2])
