import os 
import numpy as np
import matplotlib.pyplot as plt
from K_means import K_M
from sklearn import  datasets 
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler


n_samples = 500
noisy_circles = datasets.make_circles(n_samples=n_samples, factor=0.5, noise=0.05) 
noisy_moons = datasets.make_moons(n_samples=n_samples, noise=0.05)
varied = datasets.make_blobs(n_samples=n_samples, cluster_std=[1.0, 2.5, 0.5], random_state=170)
X, y = datasets.make_blobs(n_samples=n_samples, random_state=170)
transformation = [[0.6, -0.6], [-0.4, 0.8]]
X_aniso = np.dot(X, transformation)
aniso = (X_aniso, y)

default_base = {"quantile": 0.3,"eps": 0.3,"damping": 0.9,"preference": -200,"n_neighbors": 3,"n_clusters": 3,"min_samples": 7,"xi": 0.05,"min_cluster_size": 0.1,}
datasets = [(noisy_circles,{"damping": 0.77,"preference": -240,"quantile": 0.2,"n_clusters": 2,"min_samples": 7,"xi": 0.08,},),
    (noisy_moons,{"damping": 0.75,"preference": -220,"n_clusters": 2,"min_samples": 7,"xi": 0.1,},),
    (varied,{"eps": 0.18,"n_neighbors": 2,"min_samples": 7,"xi": 0.01,"min_cluster_size": 0.2,},), 
    (aniso,{"eps": 0.15,"n_neighbors": 2,"min_samples": 7,"xi": 0.1,"min_cluster_size": 0.2,},)]
L =len(datasets)


kmeans_list_info = []
for i,(dataset, algo_params) in enumerate(datasets):
    params = default_base.copy()
    params.update(algo_params)
    X, y = dataset
    X = StandardScaler().fit_transform(X)
    k_means = K_M()
    k_means_labels ,centers, video_name = k_means.fit(X,y,params["n_clusters"])
    os.rename(video_name,os.path.join("../data/output",str(i)+".avi"))
    kmeans_list_info.append((k_means_labels,centers))

fig, axs = plt.subplots(2,L,figsize=(20, 20))
fig.subplots_adjust(hspace = 0.3,wspace=0.5)
axs = axs.ravel()

for i,(dataset, algo_params) in enumerate(datasets):
    params = default_base.copy()
    params.update(algo_params)
    X, y = dataset
    X = StandardScaler().fit_transform(X)
    
    k_means_labels,centers = kmeans_list_info[i]
    axs[i].scatter(X[:,0],X[:,1],c=k_means_labels)
    axs[i].scatter(centers[:,0],centers[:,1],c="r",s=200,marker="X")
    axs[i].set_xlabel("X",fontsize=15)
    if i==0:
        axs[i].set_ylabel("K-MEANS\nRed X-points are centers  \n\n Y",fontsize=15)
    else:
        axs[i].set_ylabel("Y",fontsize=15)

    db = DBSCAN(eps=params["eps"])
    db.fit(X)
    labels = db.labels_
    inx_noise_points = np.where(labels==-1)[0]
    noise_points = X[inx_noise_points]
    X = np.delete(X, inx_noise_points,axis=0)
    labels = np.delete(labels, inx_noise_points)

    axs[i+L].scatter(X[:, 0], X[:, 1], c=labels)
    axs[i+L].scatter(noise_points[:, 0], noise_points[:, 1], c="k" )
    axs[i+L].set_xlabel("X",fontsize=15)
    if i==0:
        axs[i+L].set_ylabel("DBSCAN\nblack points are niose\n\n Y",fontsize=15)
    else:
        axs[i+L].set_ylabel("Y",fontsize=15)

plt.suptitle("Comparison K-MEANS and DBSCAN on difefrent data")
plt.show()