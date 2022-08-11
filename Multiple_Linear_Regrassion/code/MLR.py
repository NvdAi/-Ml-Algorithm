import numpy as np
import matplotlib.pyplot as plt
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
    
    def visulaze(self,X_train,Y_train):
        minimum = np.min(X_train,axis=0)
        maximum = np.max(X_train,axis=0)
        var = np.vstack((minimum,maximum))
        ones_ = np.ones(var.shape[0])
        X = np.c_[ones_,var]
        result = np.dot(X,self.thetas)
        ax = plt.axes(projection ='3d')
        points = np.c_[var,result]
        ax.plot3D(points[:,0], points[:,1], points[:,2],c="r",marker="o")
        ax.scatter3D(X_train[:,0], X_train[:,1], Y_train)
        plt.show()

    def save_model(self):
        os.makedirs("../Models", exist_ok=True)
        with open('../Models/model.pickle', 'wb') as handle:
            pickle.dump(self.thetas, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def load_model(self):
        with open('../Models/model.pickle', 'rb') as handle:
            self.thetas = pickle.load(handle)






