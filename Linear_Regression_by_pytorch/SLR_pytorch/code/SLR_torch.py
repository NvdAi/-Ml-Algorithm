import torch
import csv
import numpy as np
import pandas as pd
from torch.autograd import Variable
import matplotlib.pyplot as plt


class LinearRegressionModel(torch.nn.Module):

	def __init__(self):
		super(LinearRegressionModel, self).__init__()
		self.linear = torch.nn.Linear(1, 1) # One in and one out

	def forward(self, x):
		y_pred = self.linear(x)
		return y_pred

def visulyze(tetha_0,tetha_1,x,y):
	plt.scatter(x,y)
	mini = torch.min(x).item()
	maxi = torch.max(x).item()
	y = (tetha_0+tetha_1.item()*mini).item()
	y1 = (tetha_0+tetha_1.item()*maxi).item()
	point =[[mini,maxi],[y,y1]]
	plt.plot(point[0],point[1],c="r",marker="o",markersize=10)
	plt.xlabel(" X ",fontsize = 20,labelpad=12)
	plt.ylabel(" Y ",fontsize = 20,labelpad=12)
	plt.title("Simple Linear Regression by Pytorch")

def read_data(dirc):
	df = pd.read_csv(dirc)
	columns = df.columns.values
	x = df.loc[:,columns[1]:columns[columns.shape[0]-2]]
	y = df.loc[:,columns[-1]]
	x =  x.values.reshape(x.shape[0],x.shape[1])
	y =  y.values.reshape(y.shape[0],1)
	x_data = Variable(torch.Tensor(x))
	y_data = Variable(torch.Tensor(y))
	return x_data , y_data

x_data , y_data = read_data('../data/income.data.csv')
xp = x_data  # >> for plot;beacuse x_data must be shuffle

# define parameters
input_size, output_size = (x_data.shape[1],y_data.shape[1])
learning_rate = 0
n_epoch = 0
batch_size = 0
iteration = int(x_data.shape[0]/batch_size)

# define model
our_model = LinearRegressionModel()
criterion = torch.nn.MSELoss(size_average = True)
optimizer = torch.optim.Adam(our_model.parameters(), lr = learning_rate)

for epoch in range(n_epoch):
	for iteer in range(iteration):
		start_idx = iteer * batch_size 
		stop_idx = (iteer+1) * batch_size 
		X = x_data[start_idx:stop_idx]
		Y = y_data[start_idx:stop_idx]
		pred_y = our_model(X)
		loss = criterion(pred_y, Y)
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()
		print('epoch {}, iteration {}, loss {}'.format(epoch, iteer, loss.item()))
	print("=============================================")
	x_data = x_data[torch.randperm(x_data.size()[0])]  # shuffle data

print("================ Model trained =================")

tetha_0 = our_model.linear.bias
tetha_1 = our_model.linear.weight
print("Intereptc:",tetha_0.item(),"\nCoefficients(tetha_1):",tetha_1.item())
visulyze(tetha_0,tetha_1,xp,y_data)
plt.show()	

'''
Beta_0: 0.20427039620417498 
Beta_1: 0.713825512280208 

'''