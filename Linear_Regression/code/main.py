from linear_reg import LEANER_REGRASSION
import matplotlib.pyplot as plt
import numpy as np
import csv

def DATA_PREPARATION(dirc):
    rows = []
    with open(dirc, 'r') as csvfile:
        csvreader= csv.reader(csvfile)
        fields = next(csvreader)
        for row in csvreader:
            rows.append(row)
    data = np.array(rows)
    data = np.asarray(data, dtype=float)
    data = data[:,1:3]
    return data

dirc = "../data/income.data.csv"
data = DATA_PREPARATION(dirc)

SLR = LEANER_REGRASSION()
SLR.fit(data)
SLR.save_model()

SLR_1 = LEANER_REGRASSION()
SLR_1.load_model()
Beta_0 ,Beta_1 ,RMSE = SLR_1.evalute(data)

print("Beta_0:",Beta_0,"\nBeta_1:",Beta_1,"\nMean Squared Error:",RMSE)
plt.show()
