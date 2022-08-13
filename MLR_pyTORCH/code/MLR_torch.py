import torch
import csv
import numpy as np
import pandas as pd
from torch.autograd import Variable
import matplotlib.pyplot as plt

df = pd.read_csv('../data/income.data.csv')
columns = df.columns.values
x = df.loc[:,columns[1]]
y = df.loc[:,columns[2]]
x =  x.values.reshape(x.shape[0],1)
y =  y.values.reshape(y.shape[0],1)
x_data = Variable(torch.Tensor(x))
y_data = Variable(torch.Tensor(y))


class LinearRegressionModel(torch.nn.Module):

	def __init__(self):
		super(LinearRegressionModel, self).__init__()
		self.linear = torch.nn.Linear(1, 1) # One in and one out

	def forward(self, x):
		y_pred = self.linear(x)
		return y_pred

our_model = LinearRegressionModel()
criterion = torch.nn.MSELoss(size_average = True)
optimizer = torch.optim.SGD(our_model.parameters(), lr = 0.01)

for epoch in range(800):
	pred_y = our_model(x_data)
	loss = criterion(pred_y, y_data)
	optimizer.zero_grad()
	loss.backward()
	optimizer.step()
	print('epoch {}, loss {}'.format(epoch, loss.item()))

print("=============================")


print(our_model)
tetha_0 = our_model.linear.bias
tetha_1 = our_model.linear.weight

parametrs_list = []
for name, param in our_model.named_parameters():
	parametrs_list.append(param)


def visulyze(parametrs_list,x,y):
	plt.scatter(x,y)
	mini = np.min(x)
	maxi = np.max(x)
	y = int(parametrs_list[1]+parametrs_list[0]*mini)
	y1 = int(parametrs_list[1]+parametrs_list[0]*maxi)
	point =[[mini,maxi],[y,y1]]
	plt.plot(point[0],point[1],c="r",marker="o")

visulyze(parametrs_list,x,y)
plt.show()	