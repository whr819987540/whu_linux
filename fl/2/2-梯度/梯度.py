import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

# 求梯度

def func(X):
    """
        函数: 3*x0^2 + 5*e^x1
    """
    return 3*X[0]**2 + 5*np.exp(X[1])

def func_2(x):          #定义函数：f(x) = x1^2+x2^2
    return x[0]**2 + x[1]**2


def grandient(X:np.ndarray,func):
    """
        求偏导, 求关于x0的偏导时,和其他自变量就没有关系了
    """
    grad = np.zeros_like(X)
    for row in range(X.shape[0]):
        for column in range(X[row].shape[0]):
            # 只修改column对应的分量
            xa,xb = np.copy(X[row]),np.copy(X[row])
            xa[column] += 1e-4 
            grad[row][column] = (func(xa)-func(xb))/1e-4
    
    return grad
            

fig_1 = plt.figure()  #定义新的三维坐标轴
ax_1 = plt.axes(projection='3d')

x0 = np.arange(0.0,5.0,0.01)
x1 = np.arange(0.0,5.0,0.01)

# x0，x1是一维的，长度为n
# 相交可以得到一个n*n大小的网格，共有n*n个点
# 用x00，x11分别存储这些网格点的横纵坐标，所以x00、x11都是n*n的矩阵
x00,x11 = np.meshgrid(x0,x1)

# y也是n*n的矩阵，记录了这n*n个网格点的y坐标
y = func(np.array([x00,x11]))

ax_1.plot_surface(x00,x11,y)
plt.show()

print(grandient(np.array(list(zip(x0,x1))),func))

x0 = [3,0,3]
x1 = [4,2,0]

print(grandient(np.array(list(zip(x0,x1)),dtype=float),func_2))