import matplotlib.pyplot as plt
import numpy as np
import csv
import random

class LEANER_REGRASSION():
    def __init__(self):
        pass
    
    def PLOT(self,data,line_points):
        plt.scatter(data[:,0],data[:,1],c="r",s=20)
        plt.plot(line_points[0],line_points[1],c="b",marker="o", markersize=10)
        pass

    def evalute(self,data,Beta_0,Beta_1):
        Y_PRED = []
        for x in data[:,0]:
            Y = Beta_0+(Beta_1*x)
            Y_PRED.append(Y)
        error = data[:,1]-Y_PRED
        error = np.power(error,2)
        RMSE = np.sqrt(sum(error)/data.shape[0])    # Root Mean Squared Error
        line_points = [[np.min(data[:,0]), np.max(data[:,0])],[min(Y_PRED),max(Y_PRED)]]
        return RMSE, line_points

    def fit(self, data):
        averages = np.mean(data, axis=0)
        temp = data-averages
        UP_Division = sum(np.multiply(temp[:,0],temp[:,1]))
        DOWN_Division = sum(np.power(temp[:,0],2))
        Beta_1 = UP_Division/DOWN_Division
        Beta_0 = averages[1] - (Beta_1 * averages[0])
        RMSE, line_points = self.evalute(data,Beta_0,Beta_1)
        self.PLOT(data,line_points)
        print("Beta_0:",Beta_0,"\nBeta_1:",Beta_1,"\nMean Squared Error:",RMSE)

rows = []
with open("./income.data/income.data.csv", 'r') as csvfile:
    csvreader= csv.reader(csvfile)
    fields = next(csvreader)
    for row in csvreader:
        rows.append(row)
data = np.array(rows)
data = np.asarray(data, dtype=float)
data = data[:,1:3]
# data = np.array([[1,1],[2,3],[4,3],[3,2],[5,5]])

SLR = LEANER_REGRASSION()
SLR.fit(data)
plt.show()










