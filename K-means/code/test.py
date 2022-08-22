import matplotlib.pyplot as plt
import numpy as np

fig, axs = plt.subplots(2,3, figsize=(15, 6), facecolor='w', edgecolor='k')
fig.subplots_adjust(hspace = .5, wspace=.001)

axs = axs.ravel()

for i in range(6):
    s=np.random.rand(10,2)
    axs[i].scatter(s[:,0],s[:,1],cmap=plt.cm.Oranges)
    axs[i].set_title(str(250+i))

plt.show()