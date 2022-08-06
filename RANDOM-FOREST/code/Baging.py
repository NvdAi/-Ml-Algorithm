import collections
import numpy as np

class BAGING :
    def __init__(self, dirc, n_bg, n_attr):
        self.data = dirc
        # self.data = self.data[:10,:]
        self.shape = self.data.shape
        self.bging = [self.data]
        self.n_bg = n_bg
        self.n_attr = int((self.shape[1]-1)*(n_attr/100))
        self.attr_indx_list = []

    def MAKE_DATA(self):
        new_data_list = []
        for item in self.attr_indx_list:
            bg = self.data[:,self.shape[1]-1]
            for indx in item:
                arrtibute = self.data[:,indx]
                bg = np.vstack((bg,arrtibute))
            
            bg = np.rot90(bg, 3)
            # tup = (bg,item)
            new_data_list.append(bg)
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
        # print(self.attr_indx_list)
        N_D_L = self.MAKE_DATA()   
        return N_D_L , self.attr_indx_list


if __name__ == "__main__":
    test = BAGING("iris.txt", 4, 50)
    test.BG()

