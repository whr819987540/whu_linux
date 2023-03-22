# 原理

如果shuffle为False，dataloader在每次运行、每个epoch中加载数据的顺序都是一致的。

如果shuffle为True，且未设置seed，dataloader在每次运行时seed不同，假设第一次seed为A，则接下来epoch中数据的顺序为A1，A2，A3；第二次seed为B，数据的顺序为B1，B2，B3。

如果shuffle为True，且设置了一个固定的seed，dataloader在每次运行时seed相同，假设seed为C，则接下来epoch中数据的顺序为C1，C2，C3。

# 测试

## shuffle=false

dataloader不会修改原数据的顺序，并且保持不变。

![image-20230224155202879](https://raw.githubusercontent.com/whr819987540/pic/main/image-20230224155202879.png)

![image-20230224155649559](https://raw.githubusercontent.com/whr819987540/pic/main/image-20230224155649559.png)

再运行一次，可以看到数据没有变化，但是seed发生了变化。

![image-20230224155707081](https://raw.githubusercontent.com/whr819987540/pic/main/image-20230224155707081.png)

## shuffle=true，未设置seed

上面提到了，seed在两次运行时发生了变化A=》B，此时如果shuffle=true，显然后续数据的顺序都会发生变化。

![image-20230224155816109](https://raw.githubusercontent.com/whr819987540/pic/main/image-20230224155816109.png)



![image-20230224155916776](https://raw.githubusercontent.com/whr819987540/pic/main/image-20230224155916776.png)

## shuffle=true，固定seed

每次运行，第n轮迭代时数据的顺序都是相同的。

![image-20230224170755467](https://raw.githubusercontent.com/whr819987540/pic/main/image-20230224170755467.png)



![image-20230224170809857](https://raw.githubusercontent.com/whr819987540/pic/main/image-20230224170809857.png)




