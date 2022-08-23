
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
# plt.subplot(1,3,1)

titles = ("SVC with linear kernel",
    "LinearSVC (linear kernel)",
    "SVC with RBF kernel",
    "SVC with polynomial (degree 3) kernel",)

fig, sub = plt.subplots(2,2)
plt.subplots_adjust(wspace=0.2, hspace=0.2)
i=1
for model,ti,ax in zip(models,titles ,sub.flatten()):
    # plt.subplot(1,3,i)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_pred,y_test)
    print("Accuracy is :",acc)
    disp = DecisionBoundaryDisplay.from_estimator(model,X_train,response_method="predict",
                                                    cmap=plt.cm.coolwarm,
                                                    alpha=0.8,
                                                    ax=ax,
                                                    xlabel="X1",
                                                    ylabel="X2",)
    ax.set_title(ti)
    ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train)
    i+=1
plt.show()




# x_min, x_max = X_train[:, 0].min() - 1, X_train[:, 0].max() + 1
# y_min, y_max = X_train[:, 1].min() - 1, X_train[:, 1].max() + 1
# h = (x_max / x_min)/100
# xx, yy = np.meshgrid(np.arange(x_min, x_max, h),np.arange(y_min, y_max, h))
# # plt.subplot(1, 1, 1)
# Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
# Z = Z.reshape(xx.shape)
# # acc = accuracy_score()
# plt.contourf(xx, yy, Z, cmap=plt.cm.Paired, alpha=0.8)
# plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train)
# plt.xlabel('Sepal length')
# plt.ylabel('Sepal width')
# plt.xlim(xx.min(), xx.max())
# plt.title('SVC with linear kernel')


