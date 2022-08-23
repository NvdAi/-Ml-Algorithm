import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn import datasets

X, y_true = make_blobs(n_samples=500, centers=2,cluster_std=1,center_box=(-5,5))
eps = 0.5
min_samples = 12
db = DBSCAN(eps=eps,min_samples=min_samples)
db.fit(X)
labels = db.labels_
n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
inx_noise_points = np.where(labels==-1)[0]
noise_points = X[inx_noise_points]
XX = np.delete(X, inx_noise_points,axis=0)
labels_ = np.delete(labels, inx_noise_points)

plt.scatter(XX[:, 0], XX[:, 1], c=labels_)
plt.scatter(noise_points[:, 0], noise_points[:, 1], c="k" )
plt.title('Number of Clusters: %d' % n_clusters)
plt.xlabel("X",fontsize=15)
plt.ylabel("Y",fontsize=15)

sc = metrics.silhouette_score(X, labels)
print("Silhouette Coefficient: ",sc)
ari = metrics.adjusted_rand_score(y_true, labels)
print("Adjusted Rand Index: ",ari)
plt.show()