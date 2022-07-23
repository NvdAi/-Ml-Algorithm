from sklearn import datasets
import collections
import numpy as np
from copy import copy
from node import Node_info

class DT_TREE :
    def __init__(self, dirc):
        self.data = np.loadtxt(dirc)
        self.atrr_THRESHOLDS_SAVE = []
        self.RIGHT_NODE_LIST = []
        self.decision = {0:0, 1:0, 2:0}
        self.nodes = {}
        self.nodenumber = 0
 
    def data_partition(self,P):
        division = int(len(self.data) * P/100)
        train = self.data[:division]
        train_labels = train[:,train.shape[1]-1]
        test = self.data[division: ,]
        test_labels = test[:,train.shape[1]-1]
        return train, train_labels, test, test_labels
    
    def get_Thresholdslist_of_attrs(self,train, attr):
        Threshholds_list = []
        temp_data = train[:,attr]
        temp_data = set(temp_data)
        temp_data = sorted(temp_data)
        Threshholds_list=[]
        for i in range(0,len(temp_data)-1):
            Threshholds_list.append((temp_data[i]+temp_data[i+1])/2)   
        Threshholds_list = set(Threshholds_list)
        Threshholds_list = sorted(Threshholds_list)
        return Threshholds_list

    def gini_calculater(self, rcl):
        train_labels = rcl[0]
        del rcl[0]
        gini_list = []
        all_instance = train_labels.shape[0]
        for i,node in enumerate(rcl):
            probability_list = []
            if node.shape[0]==0:
                pass
            else:
                dictionary = dict(collections.Counter(node))
                dictionary = dict(sorted(dictionary.items()))
                for key in dictionary:
                    probability_list.append(dictionary[key]/node.shape[0])
                gini=0
                if len(probability_list)==1:
                    gini = 1-(probability_list[0]**2)
                elif len(probability_list)==2:
                    gini = 1-(probability_list[0]**2)-(probability_list[1]**2)
                else:
                    gini = 1-(probability_list[0]**2)-(probability_list[1]**2)-(probability_list[2]**2)
                gini = (node.shape[0]/all_instance)*gini
                gini_list.append(gini)
        final_gini = gini_list[0]+gini_list[1]
        return final_gini

    def find_gini(self,train,train_labels,Threshholds_list,Attr):
        gini_list = []
        for thresh in Threshholds_list:
            rowsLeft = np.where(train[:,Attr]>=thresh)[0]
            rowsRight = np.where(train[:,Attr]<thresh)[0]
            labelsLeft = train_labels[rowsLeft]
            labelsRight = train_labels[rowsRight]
            root_and_child_list = [train_labels, labelsLeft,labelsRight]
            gini =  self.gini_calculater(root_and_child_list)
            gini_list.append(gini)
        return min(gini_list) , gini_list.index(min(gini_list))

    def best_attr_threshold(self, train, train_labels):
        gini_all_attr = []
        thresh_all_attr = []
        for attr in  range(train.shape[1]-1):
            Threshholds_list = self.get_Thresholdslist_of_attrs(train,attr)
            ig, ig_indx = self.find_gini(train,train_labels,Threshholds_list,attr)
            gini_all_attr.append(ig)
            thresh_all_attr.append(Threshholds_list[ig_indx])
        thresh_indx = gini_all_attr.index(min(gini_all_attr))
        threshold = thresh_all_attr[thresh_indx]
        return thresh_indx, threshold

    def check_childs_node(self, said_nodes_indx,child_node_labels,new_said_data,train,N):
        for sid ,nod in enumerate(child_node_labels):
            NODE_SAMPELS = nod.shape[0]
            nod = dict(collections.Counter(nod))
            if len(nod)==1:
                key = list(nod.keys())
                if sid==0:
                    N.Nleft = (-1,int(key[0])) 
                else:
                    N.Nright = (-1,int(key[0])) 
            elif NODE_SAMPELS<=5:
                values = list(nod.values())
                keys = list(nod.keys())
                max_value_indx = values.index(max(values))
                key = keys[max_value_indx]
                if sid==0:
                    N.Nleft = (-1,int(key)) 
                else:
                    N.Nright = (-1,int(key))                
            else:
                if sid==1 and N.Nleft[0]==-1:
                    N.Nright = (self.nodenumber+1," ")                 
                    temp = train[said_nodes_indx[sid]]
                    self.RIGHT_NODE_LIST.append(temp)
                elif sid==1:
                    N.Nright = (self.nodenumber+2," ")                 
                    temp = train[said_nodes_indx[sid]]
                    self.RIGHT_NODE_LIST.append(temp)
                else:
                    temp_1 = train[said_nodes_indx[sid]]
                    new_said_data.append(temp_1)
                    N.Nleft = (self.nodenumber+1," ")
        self.nodes[self.nodenumber] = N 
        return new_said_data

    def fit (self,train,train_labels):
        N = Node_info()          
        N.n_attr, N.threshold = self.best_attr_threshold(train,train_labels)    
        node_Left_indx = np.where(train[:, N.n_attr]>=N.threshold)[0]
        node_Right_indx = np.where(train[:, N.n_attr]<N.threshold)[0]
        labelsLeft = train_labels[node_Left_indx]
        labelsRight = train_labels[node_Right_indx]
        said_nodes_indx = [node_Left_indx, node_Right_indx]
        child_node_labels = [labelsLeft, labelsRight]
        new_said_data = []

        NSD = self.check_childs_node(said_nodes_indx,child_node_labels,new_said_data,train,N)

        if NSD == []:
            if self.RIGHT_NODE_LIST==[]:
                print("<<<<<<<<<<<<<<<<<<<<<<<<<<<<< DECISION TREE COMPLATED  >>>>>>>>>>>>>>>>>>>>>>>>")
                return 
            NSD.append(self.RIGHT_NODE_LIST[-1])
            del self.RIGHT_NODE_LIST[-1]
        left_data = NSD[0]
        left_labels = left_data[:,left_data.shape[1]-1]
        self.nodenumber+=1
        self.fit(left_data, left_labels)
   
    def test(self,test_data):
        print("*****************************  TEST  ************************************")
        for item in test_data:
            nd=self.nodes[0]
            label = False
            while (label==False) :
                if item[nd.n_attr] >= nd.threshold:
                    division_thresh = nd.Nleft
                    if division_thresh[0]==-1:
                        self.decision[division_thresh[1]]+=1
                        label = True
                    else:
                        nd = self.nodes[division_thresh[0]]
                else:
                    division_thresh = nd.Nright
                    if division_thresh[0]==-1:
                        self.decision[division_thresh[1]]+=1
                        label = True
                    else:
                        nd = self.nodes[division_thresh[0]]

    def accuracy(self, test_labels):
        print("*****************************  EVALUATION ********************************")
        org_labels = dict(collections.Counter( test_labels))
        org_labels = dict(sorted(org_labels.items()))
        print("orginal labels :", org_labels)      
        print("Pred labels : ", self.decision)  
        acc_all = 0
        for key in org_labels:
            if self.decision[key]<org_labels[key]:
                acc = self.decision[key]/org_labels[key]
            else:
                acc = org_labels[key]/self.decision[key]
            acc_all +=acc
            print("Accuracy of label",key,"=",acc)
        acc_all = acc_all/len(org_labels)
        print("Accuracy of all data = ",acc_all)