import torch
import csv
import numpy as np
import pandas as pd
from torch.autograd import Variable
import matplotlib.pyplot as plt

df = pd.read_csv('../data/heart.data.csv')
columns = df.columns.values
print(columns)

x = df.loc[:,columns[1]:columns[columns.shape[0]-2]]
y = df.loc[:,columns[columns.shape[0]-1]]
x =  x.values.reshape(x.shape[0],x.shape[1])
y =  y.values.reshape(y.shape[0],1)
x_data = Variable(torch.Tensor(x))
y_data = Variable(torch.Tensor(y))
# print(x_data.shape)

class LinearRegressionModel(torch.nn.Module):

    def __init__(self):
        super(LinearRegressionModel, self).__init__()
        self.linear = torch.nn.Linear(2, 1) # One in and one out

    def forward(self, x):
        y_pred = self.linear(x)
        return y_pred

our_model = LinearRegressionModel()
criterion = torch.nn.MSELoss(size_average = True)
optimizer = torch.optim.Adam(our_model.parameters(), lr = 0.001)

for epoch in range(20000):
    pred_y = our_model(x_data)
    
    loss = criterion(pred_y, y_data)
    optimizer.zero_grad()
    loss.backward()
    # torch.nn.utils.clip_grad_norm_(our_model.parameters(),30)

    optimizer.step()
    print('epoch {}, loss {}'.format(epoch, loss.item()))

print("=============================")

parametrs_list = []
for name, param in our_model.named_parameters():
    p=param.detach().numpy()
    parametrs_list.append(p)

print(parametrs_list)
def visulyze(parametrs_list,x,y):
    param = parametrs_list[0][0]
    xx, yy = np.meshgrid(range(-5,80), range(-5,35))
    z = (-param[0] * xx - param[1] * yy - parametrs_list[1]) * 1. /(-1)
    ax = plt.axes(projection ='3d')
    ax.plot_surface(xx, yy, z, alpha=0.2, color='b')
    ax.scatter3D(x[:,0], x[:,1],y,c="r")
    ax.set_title("This dataset have two independed features (x1,x2)\nand one depended variable (Y)")
    ax.set_xlabel('X1',fontsize=20,labelpad=12)
    ax.set_ylabel('X2',fontsize=20,labelpad=12)
    ax.set_zlabel('Y',fontsize=20,labelpad=12)
    plt.show()


visulyze(parametrs_list,x,y)
plt.show()