import matplotlib.pylab as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

ax3 = plt.axes(projection='3d')

#定义三维数据
xx = np.arange(-5,5,0.5)
yy = np.arange(-5,5,0.5)
# https://blog.csdn.net/lllxxq141592654/article/details/81532855
# 生成网格点，相当于是用xx，yy相交，得到的新x与y坐标，对应起来就是网格点的坐标
X, Y = np.meshgrid(xx, yy) 
print(X)
print(Y)

Z = np.sin(X)+np.cos(Y)


#作图
ax3.plot_surface(X,Y,Z,cmap='rainbow')
#ax3.contour(X,Y,Z, zdim='z',offset=-2，cmap='rainbow)   #等高线图，要设置offset，为Z的最小值
plt.show()
