from Baging import BAGING
from DT import DT_TREE
import numpy as np


class RF():
    def __init__(self, n_trees, percent):
        self.n_trees = n_trees
        self.percent = percent
        self.models = []
        self.trees_pred = 0

    def fit(self, X, x):
        baging = BAGING(X, self.n_trees, self.percent)
        NEW_DATA_LIST, Features_indx_list = baging.BG()
        # print(NEW_DATA_LIST,"\n",Features_indx_list)
        # print("============================================")
        for i,data in enumerate(NEW_DATA_LIST):
            DT = DT_TREE()
            labels = data[:,data.shape[1]-1]
            DT.fit(data, labels)
            temp = (DT , Features_indx_list[i])
            self.models.append(temp)
        print("+++++++++++++++++++++++++++++ Random Forest Trained +++++++++++++++++++++++++++++++")
    
    def majority_vote(self):
        final_pred = []
        for col in range(self.trees_pred.shape[1]):
            column = self.trees_pred[:,col]
            final_label = np.bincount(column).argmax()
            final_pred.append(final_label)
        return final_pred

    def predict(self ,test_data):
        all_tree_preds = []
        for i,item in enumerate(self.models):
            bg = test_data[:,test_data.shape[1]-1]
            model = item[0]
            indexs = item[1]

            for idx in indexs:
                arrtibute = test_data[:,idx]
                bg = np.vstack((bg,arrtibute))
            new_test_data = np.rot90(bg, 3)

            print("test data for tree",i,"\n",new_test_data,"\n")
            print("like node tree",i)
            for k in model.nodes.keys():
                s = model.nodes[k]
                print(k,s.n_attr,s.threshold, s.Nleft, s.Nright)
            print("=============================================")
            
            pred_list = []
            for item in new_test_data:
                mod = model.nodes
                nd = mod[0]

                label = False
                while (label==False) :
                    if item[nd.n_attr] >= nd.threshold:
                        division_thresh = nd.Nleft
                        if division_thresh[0]==-1:
                            # self.decision[division_thresh[1]]+=1
                            pred_list.append(division_thresh[1])
                            label = True
                        else:
                            nd = mod[division_thresh[0]]
                    else:
                        division_thresh = nd.Nright
                        if division_thresh[0]==-1:
                            # self.decision[division_thresh[1]]+=1
                            pred_list.append(division_thresh[1])
                            label = True
                        else:
                            nd = mod[division_thresh[0]]
            # print(pred_list)
            all_tree_preds.append(pred_list)
        # print(all_tree_preds) 
        pred_matrix = np.array(all_tree_preds)
        self.trees_pred = pred_matrix
        print("the pred matrix is ready to take the majority vote ")
        print(self.trees_pred)
        final_pred = self.majority_vote()
        return final_pred

