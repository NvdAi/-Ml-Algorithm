from node import Node_info
from DT import DT_TREE

MODEL = DT_TREE("iris.txt")
train, train_labels, test, test_labels = MODEL.data_partition(80)
MODEL.fit(train, train_labels)
MODEL.test(test)
MODEL.accuracy(test_labels)
