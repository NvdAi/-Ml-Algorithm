import matplotlib.pyplot as plt 
import numpy as np
import random
import math

points= np.array([[3,6],[2,12],[5,8],[4,11]])
# def calculation distance points of line >>>>dis_pl
def dis_pl(points,m,c):
    Y=-1
    distance = 0
    for i in points:
        up = abs( (m*i[0]) + (Y*i[1]) + c)
        down = pow(m,2) + pow((-abs(Y)),2 ) 
        down = math.sqrt(down)
        dis = up/down
        distance += dis
    return distance    

M_ = np.random.uniform(1.0,20.0,(10,1))
C_ = np.random.uniform(1.0,20.0,(10,1))
# M_ = np.array([[3],[6],[2],[2],[4]])
# C_ = np.array([[4],[5],[5],[3],[7]])
iteration = 5    # 5 by random
distance_list =[]
m_list = []
c_list = []
for i in range(iteration):
    for ic,im in enumerate(M_):
        M = im[0]
        C = C_[ic]
        C = C[0]
        distance_list.append(dis_pl(points,M,C))

    indx = distance_list.index(min(distance_list))
    temp_m = M_[indx]
    temp_c = C_[indx]
    m = M_[indx][0]
    m_list.append(m)
    c = C_[indx][0]
    c_list.append(c)
    M_ = np.random.uniform(m-3,m+3,(9,1))
    print(M_.shape)
    M_ = np.vstack((temp_m,M_))
    C_ = np.random.uniform(c-3,c+3,(9,1))
    C_ = np.vstack((temp_c,C_))
    distance_list = []

def drow_line(m_list,c_list):
    X = np.linspace( min(points[:,0])-5, max(points[:,0])+5, 4)
    for j,i in enumerate(m_list):
        if j==len(c_list)-1:
            Y = i*X+c_list[j]
            plt.plot(X,Y,c="r")
        else:
            Y = i*X+c_list[j]
            plt.plot(X,Y,c="b")
            
drow_line(m_list,c_list)
plt.scatter(points[:,0],points[:,1],c="k",s=100)
plt.show()