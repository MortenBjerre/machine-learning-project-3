# exercise 11.1.5
from matplotlib.pyplot import figure, plot, legend, xlabel, show
import numpy as np
from scipy.io import loadmat
from sklearn.mixture import GaussianMixture
from sklearn import model_selection
from dataextraction import X, X_standard, attributeNames
from clusteringhierarchical import Z, z0,z1
import matplotlib.pyplot as plt
#from clustering-hierarchical import Z, z0, z1

# Load Matlab data file and extract variables of interest
X = X_standard
y = X[:,9]
X=X[:,:9]
N, M = X.shape

# Range of K's to try
KRange = range(1,11)
T = len(KRange)

covar_type = 'full'       # you can try out 'diag' as well
reps = 3                  # number of fits with different initalizations, best result will be kept
init_procedure = 'kmeans' # 'kmeans' or 'random'

# Allocate variables
BIC = np.zeros((T,))
AIC = np.zeros((T,))
CVE = np.zeros((T,))

# K-fold crossvalidation
CV = model_selection.KFold(n_splits=10,shuffle=True)

for t,K in enumerate(KRange):
        print('Fitting model for K={0}'.format(K))

        # Fit Gaussian mixture model
        gmm = GaussianMixture(n_components=K, covariance_type=covar_type, 
                              n_init=reps, init_params=init_procedure,
                              tol=1e-6, reg_covar=1e-6).fit(X)
        
        # Get BIC and AIC
        BIC[t,] = gmm.bic(X)
        AIC[t,] = gmm.aic(X)

        # For each crossvalidation fold
        for train_index, test_index in CV.split(X):

            # extract training and test set for current CV fold
            X_train = X[train_index]
            X_test = X[test_index]

            # Fit Gaussian mixture model to X_train
            gmm = GaussianMixture(n_components=K, covariance_type=covar_type, n_init=reps).fit(X_train)

            # compute negative log likelihood of X_test
            CVE[t] += -gmm.score_samples(X_test).sum()
            

# Plot results

figure(1); 
plot(KRange, BIC,'-*b')
plot(KRange, AIC,'-xr')
plot(KRange, 2*CVE,'-ok')
legend(['BIC', 'AIC', 'Crossvalidation'])
xlabel('K')
show()
#%%

from toolbox_02450 import clusterplot
covar_type = 'full'       # you can try out 'diag' as well
reps = 3                  # number of fits with different initalizations, best result will be kept
init_procedure = 'kmeans' # 'kmeans' or 'random'
gmm = GaussianMixture(n_components=8, covariance_type=covar_type, 
                              n_init=reps, init_params=init_procedure,
                              tol=1e-6, reg_covar=1e-6).fit(X)
clustering = gmm.predict(X)
a = clustering.reshape(clustering.shape[0],1)
plt.figure(1,figsize=(10,7))
for i in range(8):
    plt.scatter(Z[clustering == i][:,0],Z[clustering == i][:,1],label="cluster " + str(i))
legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.title("GMM clustering")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.savefig("GMM-cluster-plot.png")
plt.show()    
    
#3d
def threedplot():
    f = plt.figure()
    ax = f.add_subplot(111, projection='3d')
    for i in range(8):
        #ax.scatter(z0[:,0], z0[:,1], z0[:,2], c='g', label = "chd=0")
        #ax.scatter(z1[:,0], z1[:,1], z1[:,2], c='r', label = "chd=1")
        ax.scatter(Z[clustering == i][:,0],Z[clustering == i][:,1],Z[clustering == i][:,2], label="cluster " + str(i))
    
    
    
    plt.legend()
    plt.xlabel('PCA1')
    plt.ylabel('PCA2')
    ax.set_zlabel('PCA3')
    plt.tight_layout()
    plt.show()
