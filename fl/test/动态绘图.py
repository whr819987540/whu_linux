from math import e
from time import sleep
import matplotlib.pyplot as plt



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

def data_generator_1():
    for i in range(10):
        # sleep(2)
        yield i,i**2

def data_generator_2():
    for i in range(10):
        # sleep(2)
        yield i,i**3
        
def data_generator_3():
    for i in range(10):
        # sleep(2)
        yield i,e**i

dp = DynamicPlot(3,["i^2","i^3","e^i"])

for data1,data2,data3 in zip(data_generator_1(),data_generator_2(),data_generator_3()):
    dp.add([data1,data2,data3])
    sleep(1)

input()