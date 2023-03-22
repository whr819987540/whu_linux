from torch import nn
import torch
from torch.utils import data


# 生成数据
def generate_data(w, b, sample_number):
    x = torch.normal(0, 1, (sample_number, w.shape[0]))
    y = torch.matmul(x, w) + b + torch.normal(0, 0.01, (sample_number,))
    # 为了方便后面计算损失函数，必须将y设置为y.shape[0]*1
    return x, y.reshape(-1,1)


w = torch.tensor([2, -3.4])
b = torch.tensor([4.2])
n = 1000
features, labels = generate_data(w, b, n)

# 加载数据


def load_data(my_data, batch_size, is_train=True):
    dataset = data.TensorDataset(*my_data)
    return data.DataLoader(dataset, batch_size, shuffle=is_train)


batch_size = 10
data_loader = load_data((features, labels), batch_size, is_train=True)

# 定义模型
net = nn.Sequential(
    nn.Linear(2, 1),
)
net[0].weight.data.normal_(0, 1)
net[0].bias.data.zero_()

# 定义损失函数
loss = nn.MSELoss()

# 定义优化算法
trainer = torch.optim.SGD(net.parameters(), lr=0.03)

# 训练
epochs = 10
for epoch in range(epochs):
    for x, y in data_loader:
        y_hat = net(x)
        # print(y_hat.shape,y.shape) torch.Size([10, 1]) torch.Size([10])
        # UserWarning: Using a target size (torch.Size([10])) that is different to the input size (torch.Size([10, 1])). 
        # This will likely lead to incorrect results due to broadcasting. Please ensure they have the same size.
        l = loss(y_hat, y)
        trainer.zero_grad()
        l.backward()
        trainer.step()
    # UserWarning: Using a target size (torch.Size([1000])) that is different to the input size (torch.Size([1000, 1])). 
    # This will likely lead to incorrect results due to broadcasting. 
    # Please ensure they have the same size.
    l = loss(net(features), labels)
    print(f"epoch {epoch}, loss {l}, w {net[0].weight.data}, b {net[0].bias.data}")
