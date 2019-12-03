from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from dataextraction import X_standard, attributeNames
from sklearn.mixture import GaussianMixture
from toolbox_02450 import clusterval
import numpy as np

X = X_standard
y = X[:,9] #make y the chd column
X = X[:,:9] #get rid of chd column

covar_type = 'full'       # you can try out 'diag' as well
reps = 3                  # number of fits with different initalizations, best result will be kept
init_procedure = 'kmeans'

gmm = GaussianMixture(n_components=8, covariance_type=covar_type, 
                              n_init=reps, init_params=init_procedure,
                              tol=1e-6, reg_covar=1e-6).fit(X)
        
link = linkage(X, method="ward", metric="euclidean")
# # Compute clusters by thresholding the dendrogram
Maxclust = 8
cls = fcluster(link, criterion='maxclust', t=Maxclust)

b = gmm.predict(X)


Rand_hc, Jaccard_hc, NMI_hc = clusterval(cls,y) 
Rand_gmm, Jaccard_gmm, NMI_gmm = clusterval(b,y) 
# the exercise script 10_1_3 shows this as the output of the function 
# clusterval while help(clusterval) seems to tell something different.
print(Rand_hc, Jaccard_hc, NMI_hc)
print(Rand_gmm, Jaccard_gmm, NMI_gmm)



