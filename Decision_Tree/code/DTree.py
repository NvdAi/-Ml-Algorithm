from DT import DT_TREE
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

DATA = np.loadtxt("../dataset/iris.txt")
LABELS = DATA[:,DATA.shape[1]-1]
train_data, test_data, train_labels, test_labels = train_test_split(DATA, LABELS, train_size=0.8, test_size=0.2, random_state=None)

MODEL = DT_TREE()
MODEL.fit(train_data, train_labels)
pred_labels = MODEL.Predict( test_data)

# MODEL.save_model()
# model = DT_TREE()
# model.load_model() 
# pred_labels = model.Predict( test_data)

print("test labels :", test_labels)
print("pred_labels is :",pred_labels)
print("============================")
acc = accuracy_score(test_labels,pred_labels)
print("accuracy is : ",acc)