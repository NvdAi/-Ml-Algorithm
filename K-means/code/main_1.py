import time
import warnings
import numpy as np
import matplotlib.pyplot as plt
from sklearn import cluster, datasets, mixture
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import StandardScaler
from itertools import cycle, islice
from K_means import K_M
from sklearn.cluster import DBSCAN


n_samples = 500
noisy_circles = datasets.make_circles(n_samples=n_samples, factor=0.5, noise=0.05) 
noisy_moons = datasets.make_moons(n_samples=n_samples, noise=0.05)
blobs = datasets.make_blobs(n_samples=n_samples, random_state=8)

datasets = [noisy_circles,noisy_moons,blobs]
N_cluster = 3
fig, axs = plt.subplots(1,3, figsize=(15, 6))
fig.subplots_adjust(hspace = 0.5)
# axs = axs.ravel()

for i,dataset in enumerate(datasets):
    X, y = dataset
    X = StandardScaler().fit_transform(X)

    k_means = K_M()
    k_means_labels ,centers= k_means.fit(X,y,N_cluster)
    

    axs[i].scatter(X[:,0],X[:,1],c=k_means_labels)
    axs[i].scatter(centers[:,0],centers[:,1],c="r",s=200,marker="X")
    # plt.xlabel('X',fontsize=20,labelpad=8)
    # plt.ylabel('Y',fontsize=20,labelpad=8)

    # db = DBSCAN(eps=0.3,min_samples=7)
    # db.fit(X)
    # core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    # core_samples_mask[db.core_sample_indices_] = True
    # labels = db.labels_
    # n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    # unique_labels = set(labels)

    # colors = ['y', 'b', 'g', 'r',"c","m"]
    # for k, col in zip(unique_labels, colors):
    #     if k == -1:
    #         col = 'k'
    #     class_member_mask = (labels == k)
    #     xy = X[class_member_mask & core_samples_mask]
    #     axs[i+3].plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=col,markeredgecolor='k',markersize=6)
    #     xy = X[class_member_mask & ~core_samples_mask]
    #     axs[i+3].plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=col,markeredgecolor='k',markersize=6)
plt.show()