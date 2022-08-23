from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import numpy as np
import random
from KNN import Knn

centers = [[2, 3], [5, 5], [1, 8]]
n_classes = len(centers)
data, labels = make_blobs(n_samples=300,  centers=np.array(centers),random_state=None,cluster_std=2)
res = train_test_split(data, labels, train_size=0.8, test_size=0.2, random_state=None)
train_data, test_data, train_labels, test_labels = res 

k = 7
# scikit leartn class
scikitknn = KNeighborsClassifier(n_neighbors=k)
# my class
nvdknn = Knn(n_neighbors=k)
# compare two object
knnList = [scikitknn, nvdknn]
knnList_name = ["scikitknn", "nvdknn"]
acc_list = []
for i,knn in enumerate(knnList):
    knn.fit(train_data, train_labels) 
    predicted = knn.predict(test_data)
    acc = accuracy_score(predicted, test_labels)
    acc_list.append(acc)
    print(" the best accuracy_score of",knnList_name[i]+' =', acc)

plt.figure("KNN ALGORITHM")
plt.subplot(1,2,1)
colours = ('green', 'red', 'blue')
n_classes = 3
for n_class in range(0, n_classes):
    plt.scatter(train_data[train_labels==n_class, 0], train_data[train_labels==n_class, 1], c=colours[n_class], s=20, label=str(n_class))
    plt.scatter(test_data[test_labels==n_class, 0], test_data[test_labels==n_class, 1], c=colours[n_class], s=80)
plt.legend(loc='upper right');

plt.subplot(1,2,2)
plt.scatter(k, acc_list[0], c="b", s=50, label=knnList_name[1])
plt.scatter(k+1, acc_list[0], c="r", s=50, label=knnList_name[0])
plt.xlabel('K-Nearest Neighbors',fontsize=15)
plt.ylabel('Accuracy',fontsize=15)
plt.xlim(5, k+3)
plt.xticks(np.arange(k,k+2), ['7', '7'])
plt.legend(loc='upper right')
plt.show()
