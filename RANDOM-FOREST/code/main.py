import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from Random_Frest import RF

DATA = np.loadtxt("../dataset/iris.txt")
LABELS = DATA[:,DATA.shape[1]-1]
train_data, test_data, train_labels, test_labels = train_test_split(DATA, LABELS, train_size=0.8, test_size=0.2, random_state=None)
train_data = train_data[:100,:]
test_data = test_data[:10,:]
train_labels = train_labels[:10]
test_labels = test_labels[:10]


RF_MODEL = RF(n_trees=4,percent=50)
RF_MODEL.fit(train_data, train_labels)
pred = RF_MODEL.predict(test_data)
print("==================================================")
print("test_labes : ", test_labels,"\n","pred_labels : ",pred)
print("==================================================")
acc = accuracy_score(test_labels, pred)
print("accuracy is : ",acc)   

