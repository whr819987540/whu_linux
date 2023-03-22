import torch
import torch.nn as nn
from torchviz import make_dot,make_dot_from_trace


class DemoModel(nn.Module):
    def __init__(self):
        super(DemoModel, self).__init__()
        self.model=nn.Sequential(
            nn.Conv2d(in_channels=1,out_channels=8,kernel_size=3,padding='same'),
            nn.AvgPool2d(kernel_size=2),  # （32,1,14,14）
            nn.BatchNorm2d(num_features=8),
            nn.ReLU(),
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, padding='same'),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(in_features=14*14*16,out_features=10),
            nn.Softmax()
        )

    def forward(self,x):
        return self.model(x)


model=DemoModel()
inp_tensor=torch.ones(size=(1,1,28,28),requires_grad=True)
out=model(inp_tensor)

# 重点就是下面这两行代码
graph=make_dot(out,params=dict(model.named_parameters()),)  # 生成计算图结构表示
# graph=make_dot(out,params=dict(model.named_parameters()),show_attrs=True,show_saved=True)  # 生成计算图结构表示
# graph=make_dot(out)  # 生成计算图结构表示
# graph=make_dot(out,dict(list(model.named_parameters())+[('x',inp_tensor)]))  # 生成计算图结构表示
graph.render(filename='cnn',view=False,format='png')  # 将源码写入文件，并对图结构进行渲染
# filename：默认生成文件名为filename+'.gv'.s
# view：表示是否使用默认软件打开生成的文件
# format：表示生成文件的格式，可为pdf、png等格式
