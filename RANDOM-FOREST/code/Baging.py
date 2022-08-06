import numpy as np

class BAGING :
    def __init__(self, dirc, n_bg, n_attr):
        self.data = dirc
        self.shape = self.data.shape
        self.n_bg = n_bg
        self.n_attr = int((self.shape[1]-1)*(n_attr/100))
        self.attr_indx_list = []

    def MAKE_DATA(self):
        new_data_list = []
        labels = self.data[:,self.shape[1]-1]
        labels = np.reshape(labels,(labels.shape[0],1))
        for item in self.attr_indx_list:
            attributes = self.data[:,item]
            attributes = np.hstack((attributes,labels))
            new_data_list.append(attributes)
        return new_data_list
         
    def BG(self):
        while len(self.attr_indx_list) != self.n_bg :
            feature_index_list = []
            rep_test = -1
            while len(feature_index_list) != self.n_attr:
                F = np.random.randint(0,self.shape[1]-1)
                if F != rep_test:
                    feature_index_list.append(F)
                    rep_test=F
                else:
                    pass
            revrs = feature_index_list[::-1]
            if feature_index_list in self.attr_indx_list:
                pass
            elif revrs in self.attr_indx_list:
                pass 
            else:
                self.attr_indx_list.append(feature_index_list)
        N_D_L = self.MAKE_DATA()  
        return N_D_L , self.attr_indx_list

