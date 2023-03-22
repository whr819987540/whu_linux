#!/usr/bin/env python
# coding: utf-8

# # import
# 

# In[1]:


import torch
from torch import nn
from d2l import torch as d2l
import torchvision
from torch.utils.data import DataLoader, random_split
import torchsummary
import random
import numpy as np
from copy import deepcopy
from torch.utils.tensorboard import SummaryWriter
from os.path import join
from datetime import datetime


# # 加载数据
# 

# In[2]:


client_number = 100
seed = 0
B = 10
batch_size = 64


# In[3]:


def set_seed(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


# In[4]:


def load_data_IID(client_number, seed):
    # shuffle,fix the seed
    # 100 clients, each 100 examples
    dataset_path = "/home/whr-pc-ubuntu/code/dataset"
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.1307,), (0.3081)), # 归一化，有利于训练
    ])
    train_dataset = torchvision.datasets.MNIST(dataset_path, True, transform, download=True)
    test_dataset = torchvision.datasets.MNIST(dataset_path, False, transform, download=True)

    slice_num = int(len(train_dataset) / client_number)
    split_list = [slice_num]*(client_number-1)
    split_list.append(len(train_dataset)-sum(split_list))

    set_seed(seed)
    train_datasets = random_split(train_dataset, split_list)

    return train_datasets, test_dataset


# In[5]:


train_datasets, test_dataset = load_data_IID(client_number, seed)


# In[6]:


test_dataloader = DataLoader(test_dataset,batch_size)


# In[7]:


train_datasets[0][1][1]


# In[8]:


for i in train_datasets[0]:
    print(i[1])


# In[9]:


def load_data_Non_IDD(client_number):
    # short by digit label, ascending
    # 200 shards, each 300 examples
    # 100 clients, each 2 shards
    # that is, 100 clients, each 600 examples
    dataset_path = "/home/whr-pc-ubuntu/code/dataset"
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
    ])
    train_dataset = torchvision.datasets.MNIST(dataset_path, True, transform, download=True)
    test_dataset = torchvision.datasets.MNIST(dataset_path, False, transform, download=True)

    # before = [i[1]for i in train_dataset]
    # print(before)
    train_dataset = sorted(train_dataset, key=lambda x: x[1])
    # after = [i[1] for i in train_dataset]
    # print(after)

    slice_num = int(len(train_dataset) / client_number)
    train_datasets = []
    for i in range(client_number-1):
        train_datasets.append(train_dataset[i*slice_num:(i+1)*slice_num])
    train_datasets.append(train_dataset[(client_number-1)*slice_num:])

    return train_datasets, test_dataset


# In[10]:


train_datasets, test_dataset = load_data_Non_IDD(50)


# In[11]:


train_datasets[0][1][1]


# In[12]:


for i in train_datasets[0]:
    print(i[1])


# # 网络结构
# 

# ## MNIST_2NN
# 

# In[13]:


class MNIST_2NN(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        # 输入：784
        # 隐藏层 1：784*200，200
        # 隐藏层 2：200*200，200
        # 输出：200*10,10
        self.flat = torch.nn.Flatten()
        self.fc_1 = torch.nn.Linear(784, 200)
        self.fc_2 = torch.nn.Linear(200, 200)
        self.fc_3 = torch.nn.Linear(200, 10)
        self.relu = torch.nn.ReLU()

    def init_params(self, seed):
        set_seed(seed)
        for layer in self.children():
            if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
                # 参数初始化方法一般与激活函数有关
                # Relu-kaming
                # sigmoid-xavier
                nn.init.kaiming_normal_(layer.weight.data)
                nn.init.zeros_(layer.bias.data)

    def forward(self, x):
        x = self.flat(x)
        x = self.fc_1(x)
        x = self.relu(x)
        if self.training: # 训练模式
            x = nn.Dropout(0.5)(x)  # 过拟合
        x = self.fc_2(x)
        x = self.relu(x)
        if self.training: # 训练模式
            x = nn.Dropout(0.5)(x)  # 过拟合
        x = self.fc_3(x)
        return x


# ## CNN
# 

# In[14]:


class FedAvgCNN(nn.Module):
    def __init__(self,dropout=0.5) -> None:
        super().__init__()
        self.conv2d_1 = nn.Conv2d(1,32,(5,5))
        self.max_pool = nn.MaxPool2d((2,2))
        self.conv2d_2 = nn.Conv2d(32,64,(5,5))
        self.flat = nn.Flatten()
        self.fc = nn.Linear(1024,512)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax()
        self.linear = nn.Linear(512,10)

    def forward(self,x):
        x = self.conv2d_1(x) # 32,24,24
        x = self.relu(x) # 32,24,24
        x = self.max_pool(x) # 32,12,12

        x = self.conv2d_2(x) # 32,12,12
        x = self.relu(x) # 32,12,12
        x = self.max_pool(x) # 32,6,6

        x = self.flat(x) # 1152
        x = self.fc(x) # 512
        x = self.relu(x) # 512
        x = self.dropout(x) # 512

        x = self.softmax(x) # 512

        x = self.linear(x) # 10

        return x

    def init_params(self, seed):
        set_seed(seed)
        for layer in self.children():
            if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
                # 参数初始化方法一般与激活函数有关
                # Relu-kaming
                # sigmoid-xavier
                nn.init.kaiming_normal_(layer.weight.data)
                nn.init.zeros_(layer.bias.data)


# # global update

# In[15]:


def update_global_net(global_net, local_net, global_num, local_num):
    index = 1.0 * local_num / global_num
    optim_1 = torch.optim.SGD(global_net.parameters(), 0.1)  # whatever the lr is
    optim_2 = torch.optim.SGD(local_net.parameters(), 0.1)  # whatever the lr is

    for param_1, param_2 in zip(optim_1.param_groups[0]['params'], optim_2.param_groups[0]['params']):
        param_1.data = param_1.data + index * param_2.data


# In[16]:


def global_net_zero(global_net):
    optim = torch.optim.SGD(global_net.parameters(), 0.1)  # whatever the lr is

    for param in optim.param_groups[0]['params']:
        param.data.zero_()


# # 计算 test acc
# 

# In[17]:


def get_test_acc(net,test_dataloader,device=torch.device("cpu:0")):
    net.eval() # evaluation mode, don't use the dropout layer
    with torch.no_grad():
        sum = 0
        for x,y in test_dataloader:
            x = x.to(device)
            y = y.to(device)
            
            y_hat = net(x)
            sum += (y_hat.argmax(dim=1) == y).sum().item()
    return 1.0 * sum / len(test_dataset)
                


# In[18]:


net = MNIST_2NN()
net.init_params(seed)
get_test_acc(net, test_dataloader)


# # MNIST_2NN exp
# 

# In[19]:


seed = 0  # to initialize the global net
E = 1  # epoch
client_number = 100  # client_number
C_list = [0, 0.1, 0.2, 0.5, 1.0]  # m=max(c*client_num,1)
test_acc_target = 0.96  # when to stop the iteration
lr = 0.01


# ## IID
# 

# In[20]:


def client_update(global_net, train_dataloader, E, lr,device=torch.device("cpu:0")):
    """
        return net, loss, acc
    """
    # deep copy global_net
    local_net = deepcopy(global_net)
    loss_function = nn.CrossEntropyLoss()
    optim = torch.optim.SGD(local_net.parameters(), lr,weight_decay=0.0001,momentum=0.9) # 过拟合
    accumulator = d2l.Accumulator(3)

    local_net.train()
    for e in range(E):  # epoch E
        for x, y in train_dataloader:  # batch size B
            x = x.to(device)
            y = y.to(device)
            
            optim.zero_grad()
            y_hat = local_net(x)
            loss = loss_function(y_hat, y)
            loss.backward()
            optim.step()
            
            accumulator.add(loss*x.shape[0],d2l.accuracy(y_hat,y),x.shape[0])
            
    return local_net,accumulator[0] / accumulator[2], accumulator[1]/accumulator[2]


# In[21]:


def now_str():
    return datetime.now().strftime("%Y-%m-%d-%H-%M-%S")


# In[22]:


def get_writer(*tags):
    path = 'logs'
    for tag in tags:
        path = join(path, tag)
    writer = SummaryWriter(path)
    return writer



seed = 0  # to initialize the global net
E = 5  # epoch
client_number = 100  # client_number
C_list = [0, 0.1, 0.2, 0.5, 1.0]  # m=max(c*client_num,1)
# C_list = [1.0]  # m=max(c*client_num,1)
test_acc_target = 0.99  # when to stop the iteration
lr = 0.001


# ## IID

# In[26]:


train_datasets, test_dataset = load_data_IID(client_number, seed)
test_dataloader = DataLoader(test_dataset, 128, shuffle=False)


# In[27]:


def CNN_IID_train(C_list, E,B, lr, seed, train_datasets, test_dataloader, test_acc_target, client_number,device):
    # 加载数据，当B变化时，数据不同
    train_dataloaders = [DataLoader(train_dataset, len(train_dataset) if B == 'inf' else B,
                                    shuffle=False) for train_dataset in train_datasets]

    now_time = now_str()
    for C in C_list:
        writer = get_writer('FedAvgCNN', 'IID',f'test_acc={test_acc_target},lr={lr}',f'B={B}', now_time , f'C={C}') # divided by the start-running time
        global_net = FedAvgCNN()
        global_net.init_params(seed)
        global_net.to(device)
        step = 0
        test_acc = 0
        while test_acc < test_acc_target:  # control the variable t by the acc target
            m = max(int(C*client_number), 1)
            client_indexs = random.sample(range(0, client_number), m)  # select m clients randomly

            client_nets = []  # store net(t+1,client_index) by local update
            accumulater = d2l.Accumulator(3)
            for client_index in client_indexs:
                client_net,train_loss,train_acc = client_update(global_net, train_dataloaders[client_index], E, lr,device)
                client_nets.append(client_net)
                length = len(train_datasets[client_index]) # example number
                accumulater.add(train_loss*length,train_acc*length,length)

            global_net_zero(global_net)  # make global net's params all zero
            n = 0 # get n. n should be the sum of examples in variable client_nets, not 60000
            for client_index in client_indexs:
                n += len(train_datasets[client_index]) # example number
            for client_index in client_indexs:  # update global net
                update_global_net(global_net, client_nets[client_indexs.index(client_index)], n, len(train_datasets[client_index]))

            # check whether test acc reach the target
            test_acc = get_test_acc(global_net, test_dataloader,device)
            step += 1
            
            writer.add_scalar("train loss", accumulater[0] / accumulater[2], step)
            writer.add_scalar("train acc", accumulater[1] / accumulater[2], step)
            writer.add_scalar("test acc", test_acc, step)


# ### B=10

# In[28]:


# B = 10  # batch size for all clients
B = 'inf'  # batch size for all clients

lr = 0.001


# In[29]:


device = d2l.try_gpu()
device


# In[31]:


CNN_IID_train(C_list, E,B, lr, seed, train_datasets, test_dataloader, test_acc_target, client_number,device)


