
import matplotlib.pyplot as plt
import numpy as np
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.inspection import DecisionBoundaryDisplay


DATA = np.loadtxt("../data/iris.txt")
LABELS = DATA[:,DATA.shape[1]-1]
X_train, X_test, y_train, y_test = train_test_split(DATA, LABELS, train_size=0.9, test_size=0.1, random_state=None)
X_train = X_train[:,:2]
X_test = X_test[:,:2]

model_0 = svm.SVC(kernel="linear", C=1)
model_1 = svm.LinearSVC()
model_2 = svm.SVC(kernel="rbf", gamma=0.7, C=1)
model_3 = svm.SVC(kernel="poly", degree=3, gamma="auto", C=1)
models = [model_0,model_1,model_2,model_3]

titles = ("SVC with linear kernel","LinearSVC (linear kernel)","SVC with RBF kernel","SVC with polynomial (degree 3) kernel",)

fig, sub = plt.subplots(2,2)
plt.subplots_adjust(wspace=0.2, hspace=0.4)
plt.suptitle("comparison different SVM Algorithms")

for model,ti,ax in zip(models,titles ,sub.flatten()):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_pred,y_test)
    print("Accuracy is :",acc)
    disp = DecisionBoundaryDisplay.from_estimator(model,X_train,response_method="predict",
        cmap = plt.cm.coolwarm, alpha=0.8,ax=ax, xlabel="X1",ylabel="X2")

    ax.set_title(ti)
    ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train)

plt.show()