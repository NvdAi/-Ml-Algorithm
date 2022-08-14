import numpy as np
import matplotlib.pyplot as plt
import itertools as it
import os 
import pickle

class Multiple_Linear_Regrassion:
    def __init__(self):
        self.thetas = 0

    def fit(self,X_train,Y_train):
        '''
        general equation for MLR is : Y = theta_0 + theta_1*X1 + theta_2*X2 + ....+ theta_n*Xn
        Normal Equation for find Coefficients(thetas): theta = (X.T X)^-1 X.T Y
        '''
        if X_train.shape[0] != X_train.shape[1]:
            ones = np.ones(X_train.shape[0])
            X_train= np.c_[ones,X_train]
        else:
            pass
        step_1 = np.dot(X_train.T,X_train)
        step_2 = np.linalg.inv(step_1)
        step_3 = np.dot(step_2,X_train.T)
        step_4 = np.dot(step_3,Y_train)
        self.thetas = step_4 
        print("Your model trained:","\nIntereptc:",self.thetas[0],"\nCoefficients(theta_1 ... theta_n):",self.thetas[1:])

    def predict(self,X_test):
        ones_ = np.ones(X_test.shape[0])
        X = np.c_[ones_,X_test]
        result = np.dot(X,self.thetas)
        return result

    def accuracy(self,pred,Y_test):
        error = pred-Y_test
        error = np.power(error,2)
        RMSE = np.sqrt(sum(error)/pred.shape[0])
        return RMSE 

    def save_model(self):
        os.makedirs("../Models", exist_ok=True)
        with open('../Models/model.pickle', 'wb') as handle:
            pickle.dump(self.thetas, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def load_model(self):
        with open('../Models/model.pickle', 'rb') as handle:
            self.thetas = pickle.load(handle)

    def visulaze(self,X_train,Y_train):
        xx, yy = np.meshgrid(range(-5,80), range(-5,35))
        z = (-self.thetas[1] * xx - self.thetas[2] * yy - self.thetas[0]) * 1. /(-1)
        ax = plt.axes(projection ='3d')
        ax.plot_surface(xx, yy, z, alpha=0.2, color='b')
        ax.scatter3D(X_train[:,0], X_train[:,1], Y_train,c="r")
        ax.set_title("This dataset have two independed features (x1,x2)\nand one depended variable (Y)")
        ax.set_xlabel('X1',fontsize=20,labelpad=12)
        ax.set_ylabel('X2',fontsize=20,labelpad=12)
        ax.set_zlabel('Y',fontsize=20,labelpad=12)
        plt.show()

