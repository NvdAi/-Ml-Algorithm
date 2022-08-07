from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from Random_Forest  import RF
import numpy as np

acc_list = []
for i in range(10):
    DATA = np.loadtxt("../dataset/iris.txt")
    LABELS = DATA[:,DATA.shape[1]-1]
    train_data, test_data, train_labels, test_labels = train_test_split(DATA, LABELS, train_size=0.8, test_size=0.2, random_state=None)


    RF_MODEL = RF(80)
    RF_MODEL.fit(train_data, train_labels)
    RF_MODEL.save_model()
    pred = RF_MODEL.predict(test_data)
    # loaded_model = RF()
    # loaded_model.load_model()
    # pred = loaded_model.predict(test_data)

    # print("==================================================")
    # print("test_labes : ", test_labels,"\n","pred_labels : ",pred)
    # print("==================================================")
    acc = accuracy_score(test_labels, pred)
    acc_list.append(acc)
    print("accuracy is : ",acc)   
average = sum(acc_list)/len(acc_list)
print("average accuracy is :",average)



