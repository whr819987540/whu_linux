from turtle import shape
import matplotlib.pyplot as plt
import numpy as np
import torch
import random


def data(w:torch.Tensor,b,sample_number):
    x = torch.normal(0,1,(sample_number,w.shape[0]))
    y = torch.matmul(x,w) + b + torch.normal(0,0.01,(sample_number,))
    return x,y

def get_sample(features,labels,batch_size):
    indices = list(range(0,len(labels)))
    random.shuffle(indices) # 打乱下标顺序，直接对源数据进行修改
    for i in range(0,len(labels),batch_size):
        index = indices[i:min(i+batch_size,len(labels))]
        yield features[index],labels[index]

w = torch.tensor([2,-3.4])
b = 4.2
features,labels = data(w,b,1000)


batch_size = 10
w = torch.normal(0,0.01,(2,),requires_grad=True)
b = torch.normal(0,0.01,(1,),requires_grad=True)


def linear_regression(x,w,b):
    return torch.matmul(x,w)+b


def square_loss_function(y_hat,y):
    return 0.5*(y_hat-y)**2

def sgd(params,lr,batch_size):
    with torch.no_grad():
        for param in params:
            # param = param - lr/batch_size*param.grad 会导致 param.grad消失
            # https://blog.csdn.net/yinizhilianlove/article/details/104033592
            param -=  lr/batch_size*param.grad 
            param.grad.zero_()
lr = 0.03 # 学习率
epochs = 10 # 学习轮数
net = linear_regression # 网络结构，这里就是一个线性回归模型
loss = square_loss_function # 损失函数
batch_size

for epoch in range(epochs):
    for X,y in get_sample(features,labels,batch_size):
        y_hat = net(X,w,b)
        l = loss(y_hat,y)
        l.sum().backward() # 反向传播，然后才能计算w、b的梯度
        sgd([w,b],lr,batch_size)
    with torch.no_grad():
        l = loss(net(features,w,b),labels)
        print(f"epoch:{epoch}, loss: {l.mean()}, w: {w[0],w[1]}, b: {b[0]}")
        
        