import matplotlib.pyplot as plt
import numpy as np
import pickle
import csv
import random
import os

class LEANER_REGRASSION():
    def __init__(self):
        self.parametrs = 0
    
    def PLOT(self,data,line_points):
        plt.scatter(data[:,0],data[:,1],c="r",s=20)
        plt.plot(line_points[0],line_points[1],c="b",marker="o", markersize=10)

    def evalute(self,data):
        Y_PRED = []
        Beta_0 ,Beta_1 = self.parametrs
        for x in data[:,0]:
            Y = Beta_0+(Beta_1*x)
            Y_PRED.append(Y)
        error = data[:,1]-Y_PRED
        error = np.power(error,2)
        RMSE = np.sqrt(sum(error)/data.shape[0])    # Root Mean Squared Error
        line_points = [[np.min(data[:,0]), np.max(data[:,0])],[min(Y_PRED),max(Y_PRED)]]
        self.PLOT(data,line_points)
        return Beta_0 ,Beta_1 ,RMSE

    def fit(self, data):
        averages = np.mean(data, axis=0)
        temp = data-averages
        UP_Division = sum(np.multiply(temp[:,0],temp[:,1]))
        DOWN_Division = sum(np.power(temp[:,0],2))
        Beta_1 = UP_Division/DOWN_Division
        Beta_0 = averages[1] - (Beta_1 * averages[0])
        self.parametrs = [Beta_0,Beta_1]
    
    def save_model(self):
        os.makedirs("../Models", exist_ok=True)
        with open('../Models/model.pickle', 'wb') as handle:
            pickle.dump(self.parametrs, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    def load_model(self):
        with open('../Models/model.pickle', 'rb') as handle:
            self.parametrs = pickle.load(handle)
    










