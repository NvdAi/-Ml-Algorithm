import numpy as np
import itertools as it

class BAGING :
    def __init__(self, dirc, n_attr):
        self.data = dirc
        self.shape = self.data.shape
        self.n_attr = int((self.shape[1]-1)*(n_attr/100))
        self.attr_indx_list = 0

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
        x = np.arange(self.shape[1]-1)
        self.attr_indx_list = list(it.combinations(x, self.n_attr))
        N_D_L = self.MAKE_DATA()
        return N_D_L , self.attr_indx_list

if __name__ == "__main__":
    test = BAGING("iris.txt", 50)
    test.BG()
