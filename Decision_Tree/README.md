Decision Tree:
Decision Tree algorithm belongs to the family of supervised learning algorithms. Unlike other supervised learning algorithms, the decision tree algorithm can be used for solving regression and classification problems too.
In Decision Trees, for predicting a class label for a record we start from the root of the tree. We compare the values of the root attribute with the record’s attribute. On the basis of comparison, we follow the branch corresponding to that value and jump to the next node.

Iris dataset:
I implemented a decision tree on iris detaset, iris is continuous variable dataset. iris have 4 features (4 col) and one column as a labela in three class, and also we have 150 samples in this dataset.

Description of my code:
0-Dtat preparation >> i used iris as numpy in txt file; so to use this code you shuold change the dataset format into numpy in txt file.

ponit* : we should find the best attribute and best division threshold to divide data to two sub tree >> to do this goal, do the following steps:

1- make thrasholds list >> i used "get_Thresholdslist_of_attrs" funqtion to get list of threshholds from each features, each thresh is mean of two neighbors sample in that column
2- Now we have to divide the data of the value of the same feature that we made from that treshold list , for each thresh in the list create the left and right subtree once
3- for each datapartition to subtres  we have to calculate "gini_calculater" the gini impurity >> for each threshold we have a gini value
4- well so far >> we have a list of threshold for each attributr/ a list of gini value corresponding to each threshold list
5- you should choose the minimum gini of gini valuse list and its corresponding threshold >> do this step for each attribute 
6- we have a single treshold (best threshold) for each feature and also we have gigi value about each them.
7- to find the best attribute to divide data you have to choose minimum gini value between 4 ginis(iris have 4 attribute)
8- now ; you have the best gini of all thresholds and all features, and also its corresponding treshhold 
9- in this step divide data by best treshold on best attribute to subtrees (left and right nodes)
10- for each node ,check it   if node is pure return sample labels  else  reoeat the all ateps for now data in each node


I have 3 codes   DT.py is my decision tree structure      node.py id my node info calss     Dtree.py is main for run.


