import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.datasets import make_blobs
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from K_means import K_M


X,org_labels = make_blobs(1200, n_features=2, centers=6, cluster_std=0.85, random_state=0)
K = 6   # number of clusters
MODEL = K_M()
clusters,centers = MODEL.fit(X, org_labels, K)

plt.subplot(1,2,1)
plt.scatter(X[:,0],X[:,1],c=clusters)
plt.scatter(centers[:,0],centers[:,1],c="r",s=200,marker="X")
plt.xlabel('X',fontsize=20,labelpad=12)
plt.ylabel('Y',fontsize=20,labelpad=12)
print("=================== k-means completed ===========================")



eps = 0.5
min_samples = 10
db = DBSCAN(eps=eps, min_samples=min_samples).fit(X)
core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
core_samples_mask[db.core_sample_indices_] = True
labels = db.labels_
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)

unique_labels = set(labels)
colors = ['y', 'b', 'g', 'r',"c","m"]
plt.subplot(1,2,2)
for k, col in zip(unique_labels, colors):
    if k == -1:
        col = 'k'
 
    class_member_mask = (labels == k)
 
    xy = X[class_member_mask & core_samples_mask]
    plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=col,
            markeredgecolor='k',
            markersize=6)
 
    xy = X[class_member_mask & ~core_samples_mask]
    plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=col,
             markeredgecolor='k',
             markersize=6)

plt.title('number of clusters: %d' % n_clusters_)
sc = metrics.silhouette_score(X, labels)
print("Silhouette Coefficient: ",sc)
ari = metrics.adjusted_rand_score(org_labels, labels)
print("Adjusted Rand Index: ",ari)
plt.show()