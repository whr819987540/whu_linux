from cmath import e
from time import sleep
import torchvision
from torch.utils import data
import torch
from matplotlib import pyplot as plt    
from IPython import display
from d2l import torch as d2l


def load_data_fashion_mnist(batch_size):
    """
        1、加载fashion mnist数据库
        
        2、将PIL格式的数据转成tensor
        
        3、将数据集按照batch_size转成可迭代的对象
        
        4、返回训练数据集与测试数据集
    """
    trans = torchvision.transforms.ToTensor()
    train_data = torchvision.datasets.FashionMNIST(
        root=r"C:\Users\user\Desktop\FL\code\data",
        train=True,
        transform=trans
    )
    test_data = torchvision.datasets.FashionMNIST(
        root=r"C:\Users\user\Desktop\FL\code\data",
        train=False,
        transform=trans
    )
    
    train_data_loader = data.DataLoader(
        dataset=train_data,
        batch_size=batch_size,
        shuffle=True,
    )
    test_data_loader = data.DataLoader(
        dataset=test_data,
        batch_size=batch_size,
        shuffle=True,
    )
    
    return train_data_loader,test_data_loader

def Test_load_data_fashion_mnist():
    batch_size = 256
    train_data_loader,test_data_loader = load_data_fashion_mnist(batch_size)
    tmp_x,tmp_y = next(iter(train_data_loader))
    print(tmp_x.shape,tmp_y.shape)

def ReLU(x):
    return torch.max(x,torch.zeros_like(x))

def Test_ReLU():
    tmp = torch.normal(0,1,(3,4))
    tmp_relu = ReLU(tmp)
    print(tmp)
    print(tmp_relu)

def accuracy(y_hat,y):
    """
        返回正确的预测数, 样本总数
    """
    return int((y_hat.max(dim=1).indices == y).sum()), y.shape[0]

def TEST_accuracy():
    y_hat = torch.normal(0,1,(4,5))
    y = torch.tensor([4,2,1,4])
    print(y_hat)
    print(y)
    print(accuracy(y_hat,y))
    
class DynamicPlot:
    """
        深度学习中, 绘图数据(模型中间结果)的生成可能会经过很长时间
        在等待最终结果的过程中, 希望每生成一组绘图数据就绘制一个图形
        这就需要动态绘图
        实现思路是, 用一个数组记录历史结果, 每生成一组新的绘图数据, 就更新历史结果
        同时, 将原来的绘图清空并重新绘图
        重新绘图是有必要的, 因为在后续过程中, 坐标轴等会发生伸缩等变化, 所以需要重新绘制
        坐标轴是很难提前确定的, 这个值会随新绘图数据的变化而变化
    """
    def __init__(self,n,labels):
        """
            n控制绘制的曲线数目
            labels控制各个曲线的图例
        """
        # data like [[x1,y1],[x2,y2],...[xn,yn]]
        # xi, yi list
        self.data = [ [[],[]] for i in range(n)]
        self.labels = labels
        self.fig,self.ax = plt.subplots()
        print(self.data)
        
    def add(self,data_matrix):
        """
            data_matrix:[[x1,y1],[x2,y2],...[xn,yn]]
            然后自动更新图形
        """
        self.ax.cla() # 清空之前绘制的图
        print(data_matrix)
        for index, data in enumerate(data_matrix):
            self.data[index][0].append(data[0]) # x
            self.data[index][1].append(data[1]) # y
            self.ax.plot(self.data[index][0],self.data[index][1],label=self.labels[index])
        self.ax.legend() # 显示图例
        plt.pause(0.1)

def TEST_DynamicPlot():
    dp = DynamicPlot(3,["i^2","i^3","e^i"])
    for i in range(10):
        a,b,c = i**2,i**3,e**i
        dp.add([[i,a],[i,b],[i,c]])
        sleep(1)       

class Animator:  #@save
    """在动画中绘制数据"""
    def __init__(self, xlabel=None, ylabel=None, legend=None, xlim=None,
                 ylim=None, xscale='linear', yscale='linear',
                 fmts=('-', 'm--', 'g-.', 'r:'), nrows=1, ncols=1,
                 figsize=(3.5, 2.5)):
        # 增量地绘制多条线
        if legend is None:
            legend = []
        d2l.use_svg_display()
        self.fig, self.axes = d2l.plt.subplots(nrows, ncols, figsize=figsize)
        if nrows * ncols == 1:
            self.axes = [self.axes, ]
        # 使用lambda函数捕获参数
        self.config_axes = lambda: d2l.set_axes(
            self.axes[0], xlabel, ylabel, xlim, ylim, xscale, yscale, legend)
        self.X, self.Y, self.fmts = None, None, fmts

    def add(self, x, y):
        # 向图表中添加多个数据点
        if not hasattr(y, "__len__"):
            y = [y]
        n = len(y)
        
        if not hasattr(x, "__len__"):
            x = [x] * n
        
        if not self.X:
            self.X = [[] for _ in range(n)]
        if not self.Y:
            self.Y = [[] for _ in range(n)]
        
        for i, (a, b) in enumerate(zip(x, y)):
            if a is not None and b is not None:
                self.X[i].append(a)
                self.Y[i].append(b)
                
        self.axes[0].cla()
        
        for x, y, fmt in zip(self.X, self.Y, self.fmts):
            self.axes[0].plot(x, y, fmt)
        self.config_axes()
        display.display(self.fig)
        display.clear_output(wait=True)

def TEST_Aminator():
    animator = Animator(xlabel="x",ylabel="f(x)",legend=["$i^2$","$i^3$","$e^i$"])
    for i in range(10):
        a,b,c = i**2,i**3,e**i
        animator.add(i,(a,b,c))
        sleep(1)  

class Accumulator:
    """
        累加器, 将中间结果统一求和
    """
    def __init__(self,n):
        """
            n表示要存放的元素类别
        """
        self.data = torch.zeros(n,dtype=torch.float32)
    
    def add(self,*args):
        """
            将args顺序放入
        """
        for i,arg in enumerate(args):
            self.data[i] += arg
    
    def reset(self):
        """
            将数值全部清空
        """
        self.data.zero_()
        
    def __getitem__(self,i):
        return float(self.data[i])
    
def TEST_Accumulator():
    accumulator = Accumulator(2)
    accumulator.add(2,4)
    accumulator.add(3,6)
    print(accumulator[0],accumulator[1])
    


    

if __name__ == "__main__":
    # Test_load_data_fashion_mnist()
    # Test_ReLU()
    # TEST_accuracy()
    # TEST_DynamicPlot()
    # TEST_Aminator()
    TEST_Accumulator()