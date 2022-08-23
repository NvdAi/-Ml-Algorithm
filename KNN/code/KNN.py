
import numpy as np 
import collections

class Knn:
    def __init__(self, n_neighbors):
        self.n_neighbors = n_neighbors

    def fit(self, X, Y):
        self.X = X
        self.Y = Y
        return self.X , self.Y

    def predict(self, x):
        predicted_labels = []
        for point in x:
            dis = list(np.sqrt(np.sum((self.X - point)**2, axis=1) ))
            dis_sorted = sorted(dis)
            min_distances = dis_sorted[0:self.n_neighbors]
            LNP = []
            for i in min_distances:
                inx = dis.index(i)
                label_of_each_near_point = int(self.Y[inx])
                LNP.append(label_of_each_near_point)
            frequency = collections.Counter(LNP)
            keys = list(frequency.keys())
            vals = list(frequency.values())
            m_freq = max(frequency.values())
            position = vals.index(m_freq)
            pred_label = keys[position]
            predicted_labels.append(pred_label)
            LNP = []
        return predicted_labels


