import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import torch

def data(w,b,sample_number):
    x = torch.normal(0,1,(sample_number,w.shape[0]))
    y = torch.matmul(x,w) + b + torch.normal(0,0.01,(sample_number,))
    print(x,y)
    print(x.shape,y.shape)
    return x,y

w = torch.tensor([2,-3.4])
b = 4.2
n = 1000
features,labels = data(w,b,n)

ax = plt.axes(projection='3d')
ax.scatter3D(features[:,0], features[:,1], labels, cmap='Greens')

plt.show()