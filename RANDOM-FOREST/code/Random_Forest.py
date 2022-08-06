from Baging import BAGING
from DT import DT_TREE
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
import pickle
import os


class RF():
    def __init__(self, n_trees, percent):
        self.n_trees = n_trees
        self.percent = percent
        self.models = []
        self.trees_pred = 0

    def fit(self, X, x):
        baging = BAGING(X, self.n_trees, self.percent)
        NEW_DATA_LIST, Features_indx_list = baging.BG()
        for i,data in enumerate(NEW_DATA_LIST):
            DT = DT_TREE()
            labels = data[:,data.shape[1]-1]
            DT.fit(data, labels)
            temp = (DT , Features_indx_list[i])
            self.models.append(temp)
    
    def majority_vote(self):
        final_pred = []
        for col in range(self.trees_pred.shape[1]):
            column = self.trees_pred[:,col]
            final_label = np.bincount(column).argmax()
            final_pred.append(final_label)
        return final_pred

    def predict(self ,test_data):
        all_tree_preds = []
        for item in self.models:
            model = item[0]
            features_indexs = item[1]
            new_test_data = test_data[:,features_indexs]
            
            # print("test data for tree","\n",new_test_data,"\n")
            # print("like node tree")
            # for k in model.nodes.keys():
                # s = model.nodes[k]
                # print(k,s.n_attr,s.threshold, s.Nleft, s.Nright)
            # print("=============================================")
            prediction = model.Predict(new_test_data)
            all_tree_preds.append(prediction)
        self.trees_pred = np.array(all_tree_preds)
        final_prediction = self.majority_vote()
        return final_prediction

    def save_model(self):
        os.makedirs("../Models", exist_ok=True)
        with open('../Models/model.pickle', 'wb') as handle:
            pickle.dump(self.models, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    def load_model(self):
        with open('../Models/model.pickle', 'rb') as handle:
            model = pickle.load(handle)
        return model
