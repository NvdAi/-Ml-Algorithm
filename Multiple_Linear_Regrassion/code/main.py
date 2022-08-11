from MLR import Multiple_Linear_Regrassion
import matplotlib.pyplot as plt
import numpy as np
import csv

def DATA_PREPARATION(dirc,partition):
    rows = []
    with open(dirc, 'r') as csvfile:
        csvreader= csv.reader(csvfile)
        fields = next(csvreader)
        for row in csvreader:
            rows.append(row)
    data = np.array(rows)
    data = np.asarray(data, dtype=float)
    data_shape = data.shape
    data = data[:,1:data_shape[1]]
    data_shape = data.shape
    row = int(data_shape[0]*(partition/100))
    X_train = data[:row, 0:data_shape[1]-1]
    Y_train = data[:row, data_shape[1]-1]
    X_test = data[row: , 0:data_shape[1]-1]
    Y_test = data[row:,data_shape[1]-1]
    return X_train,Y_train,X_test,Y_test

dirc = "../data/heart.data.csv"
X_train,Y_train,X_test,Y_test = DATA_PREPARATION(dirc,partition=97)


MLR = Multiple_Linear_Regrassion()
MLR.fit(X_train,Y_train)
MLR.save_model()
print("========================================================")
MLR_1 = Multiple_Linear_Regrassion()
MLR_1.load_model()
prediction = MLR_1.predict(X_test)
# print("prediction:\n",prediction,"\n\norginal Y:\n",Y_test)
print("========================================================")
RMSE = MLR_1.accuracy(prediction,Y_test)  # Root Mean Squared Error
print("RMSE:",RMSE)
MLR_1.visulaze(X_train,Y_train)











