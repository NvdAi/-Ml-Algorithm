from node import Node_info
from DT import DT_TREE

MODEL = DT_TREE("iris.txt")
train, train_labels, test, test_labels = MODEL.data_partition(80)
MODEL.fit(train, train_labels)
PRED_LIST = MODEL.Predict(test)
print("pred_list is :",PRED_LIST)
MODEL.accuracy(test_labels)
