
import matplotlib.pyplot as plt
import numpy as np
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.inspection import DecisionBoundaryDisplay


DATA = np.loadtxt("../data/iris.txt")
LABELS = DATA[:,DATA.shape[1]-1]
X_train, X_test, y_train, y_test = train_test_split(DATA, LABELS, train_size=0.9, test_size=0.1, random_state=None)
X_train = train_data[:,:2]
X_test = test_data[:,:2]

model = svm.LinearSVC()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
acc = accuracy_score(y_pred,y_test)
print("Accuracy is :",acc)

disp = DecisionBoundaryDisplay.from_estimator(model,X_train,response_method="predict",
                                                cmap=plt.cm.coolwarm,
                                                alpha=0.8,
                                                xlabel="X1",
                                                ylabel="X3",)
plt.title("Multiple Classification\nSupport Vector Machine (SVM)")
plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train)
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


