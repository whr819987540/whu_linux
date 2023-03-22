#求函数的梯度，也就是所有偏导的组合
import numpy as np
def numberical_grandient(f, x):
    h = 1e-4                     #定义一个微小量，不能太小，太小计算机没法正确表示
    grad = np.zeros_like(x)      #生成和x形状相同的数组
    for idx in range(x.size):    #计算所有偏导
        tmp_val = x[idx]
        x[idx] = tmp_val + h            #要计算的那个自变量加h，其余不变
        fxh1 = f(x)                     #计算f(x+h)

        x[idx] = tmp_val - h           #计算f(x-h)
        fxh2 = f(x)

        grad[idx] = (fxh1 - fxh2) / (2*h)    #计算偏导
        x[idx] = tmp_val
    return grad
def function_2(x):          #定义函数：f(x) = x1^2+x2^2
    return x[0]**2 + x[1]**2
#输出三个梯度
print(numberical_grandient(function_2, np.array([3.0, 4.0])))
print(numberical_grandient(function_2, np.array([0.0, 2.0])))
print(numberical_grandient(function_2, np.array([3.0, 0.0])))
