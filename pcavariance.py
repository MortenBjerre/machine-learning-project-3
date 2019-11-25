import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import dataextraction
from scipy.linalg import svd

# 0: sbp, 1: tobacco, 2: ldl, 3: adiposity, 4: famhist, 5: typea
# 6: obesity, 7: alcohol, 8: age, 9: chd

X = dataextraction.X
N, M = X.shape

Y = X - np.ones((N,1))*X.mean(axis=0)
# for r in range(len(Y)):
#     for c in range(len(Y[r])):
#         Y[r][c] /= np.std(Y[:,c])
Y[:,4] = X[:,4]
Y[:,9] = X[:,9]



# PCA by computing SVD of Y
U,S,V = svd(Y,full_matrices=False)

# Compute variance explained by principal components
rho = (S*S) / (S*S).sum() 

cumulrho = [] #The cumulative error explained by the PCA
for i in range(len(rho)):
    cumulrho += [np.sum(rho[0:i])]

threshold = 0.85

if __name__=="__main__":
    plt.scatter(x=np.arange(len(rho)), y=rho, c='blue', label='variance')
    plt.scatter(x=np.arange(len(rho)), y=cumulrho, c='green', label='cumulative variance')
    plt.plot(np.arange(len(rho)), threshold*np.ones(len(rho)), 'r-', label="threshold")

    plt.title('Variance explained by principal components')
    plt.xlabel('Principal component')
    plt.ylabel('Variance explained')
    plt.xticks(np.arange(len(rho)))
    plt.yticks(np.arange(11)/10.0)
    plt.legend()
    plt.grid()
    plt.show()
