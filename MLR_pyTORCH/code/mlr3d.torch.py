import torch
import csv
import numpy as np
import pandas as pd
from torch.autograd import Variable
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation
from matplotlib import cm

 
class LinearRegressionModel(torch.nn.Module):

	def __init__(self):
		super(LinearRegressionModel, self).__init__()
		self.linear = torch.nn.Linear(2, 1) # One in and one out

	def forward(self, x):
		y_pred = self.linear(x)
		return y_pred

def visualize(point_lists,xp,y_data):
	ax = plt.axes(projection ='3d')
	for i in range(len(point_lists[0])):
		if i==len(point_lists[0])-1:
			ax.plot_surface(point_lists[0][i], point_lists[1][i], point_lists[2][i], alpha=0.5, color='b')
		else:
			ax.plot_surface(point_lists[0][i], point_lists[1][i], point_lists[2][i], alpha=0.2, color='y')
	ax.scatter3D(xp[:,0], xp[:,1],y_data,c="r")
	ax.set_title("This dataset have two independed features (x1,x2)\nand one depended variable (Y)")
	ax.set_xlabel('X1',fontsize=20,labelpad=12)
	ax.set_ylabel('X2',fontsize=20,labelpad=12)
	ax.set_zlabel('Y',fontsize=20,labelpad=12)

def drow_surface(tetha_0, tetha_12, point_lists):
	xx, yy = np.meshgrid(range(-5,80), range(-5,35))
	z = (-tetha_12[0] * xx - tetha_12[1] * yy - tetha_0) * 1. /(-1)
	point_lists[0].append(xx)
	point_lists[1].append(yy)
	point_lists[2].append(z)

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

x_data , y_data = read_data('../data/heart.data.csv')
xp = x_data # >> for plot;beacuse x_data must be shuffle

# define parameters
input_size, output_size = (x_data.shape[1],y_data.shape[1])
learning_rate = 0.01
n_epoch = 200
batch_size = 25
EIO = 50 # Every 100 itreation once plot
iteration = int(x_data.shape[0]/batch_size)

#define model
our_model = LinearRegressionModel()
criterion = torch.nn.MSELoss(size_average = True)
optimizer = torch.optim.Adam(our_model.parameters(), lr = learning_rate)

point_lists = [[],[],[]]
for epoch in range(n_epoch):
	start,stop = (0,0)
	for iteer in range(iteration):
		X = x_data[0+start:batch_size+stop]
		Y = y_data[0+start:batch_size+stop]
		pred_y = our_model(X)
		loss = criterion(pred_y, Y)
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()
		# print('epoch {}, iteration {}, loss {}'.format(epoch, iteer, loss.item()))
	x_data = x_data[torch.randperm(x_data.size()[0])]  # shuffle data
	# print("=============================================")
	if epoch %EIO==0 or epoch==n_epoch-1:
		print("epoc:",epoch)	
		tetha_0 = our_model.linear.bias.tolist()[0]
		tetha_12 = our_model.linear.weight.tolist()[0]
		print("Intereptc:",tetha_0,"\nCoefficients(tetha_1....tetha2):",tetha_12)
		drow_surface(tetha_0, tetha_12, point_lists)
	# print("=============================================")

visualize(point_lists,xp,y_data)

plt.show()
'''
Intereptc: 15.01642374536851 
Coefficients(theta_1 ... theta_n): [-0.20059503  0.17741778]
'''