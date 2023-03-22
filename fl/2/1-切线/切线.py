import matplotlib.pyplot as plt
import numpy as np

# 绘制x^3-1/x的图像，在x=1处的切线图像


def line(x, y, slope, X):
    """
        根据坐标与点返回直线的y
    """
    return slope*(X-x) + y


def func(x):
    """
        待求导的函数
    """
    return x**3-1/x


def derivation(func, x, limit=1e-4):
    """
        求导
    """
    return (func(x+limit) - func(x))/limit


# x = np.arange(0,3,0.01) # 0不行，0.0可以
X = np.arange(0.0, 3, 0.01)
Y = func(X)

x_p = 1
p_slope = derivation(func, x_p)
y_p = func(x_p)


plt.plot(X, Y)
plt.plot(X,line(x_p,y_p,p_slope,X))
plt.show()
