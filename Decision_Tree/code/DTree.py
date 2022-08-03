from DT import DT_TREE
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

DATA = np.loadtxt("../dataset/iris.txt")
LABELS = DATA[:,DATA.shape[1]-1]
train_data, test_data, train_labels, test_labels = train_test_split(DATA, LABELS, train_size=0.8, test_size=0.2, random_state=None)

MODEL = DT_TREE()
MODEL.fit(train_data, train_labels)
MODEL.save_model()

dt = DT_TREE()
model = dt.load_model() 
p = model.item().keys()
print("===========node info============")
for i in p:
        s=model.item().get(i)
        print(i,"==",s.threshold,
        s.n_attr,
        s.Nleft,
        s.Nright)
print("=========== test data ============")
print(test_data)
pred_labels = dt.Predict(model, test_data)
print("=========== pred and acc by me ============")
print("test labels :", test_labels)
print("pred_labels is :",pred_labels)
dt.accuracy(test_labels)
print("=================================")
acc = accuracy_score(test_labels,pred_labels)
print("accuracy by skl ",acc)