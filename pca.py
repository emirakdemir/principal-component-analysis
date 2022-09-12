import numpy as np

rng = np.random.RandomState(1)
X = np.dot(rng.rand(2,2), rng.rand(2,200)).T
X.shape

import matplotlib.pyplot as plt
%matplotlib inline
plt.scatter(X[:,0],X[:,1])

from sklearn.decomposition import PCA
pca = PCA(n_components=2)
pca.fit(X)

pca.components_
pca.explained_variance_

pca = PCA(n_components=1)
pca.fit(X)
X_pca = pca.transform(X)
X.shape
X_pca.shape

X_new = pca.inverse_transform(X_pca)
plt.scatter(X[:,0], X[:,1], alpha=0.2)
plt.scatter(X_new[:,0],X_new[:,1], alpha=0.8)


from sklearn.datasets import load_digits
digits = load_digits()
digits.data.shape

pca = PCA(2)
data_pca = pca.fit_transform(digits.data)
data_pca.shape

pca = PCA().fit(digits.data)
plt.plot(np.cumsum(pca.explained_variance_ratio_))
