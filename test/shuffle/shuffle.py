import random
import torch
import numpy as np
from torch.utils.data import DataLoader
import os

dataset_path = './dataset.txt'
seed = 59


class Dateset:

    def __init__(self):
        if os.path.exists(dataset_path):
            self.data = np.loadtxt(dataset_path, int, delimiter=',')
        else:
            self.data = np.random.random((6, 2))*100
            self.data = self.data.astype(int)
            print(self.data)
            np.savetxt(dataset_path, self.data, fmt='%d', delimiter=',')

    def __getitem__(self, index):
        return self.data[index][0], self.data[index][1]

    def __len__(self):
        return self.data.shape[0]


dataset = Dateset()
print(len(dataset))
print(dataset[0])

print(torch.seed()) # 设置seed的同时返回一个该seed

torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)

# https://pytorch.org/docs/stable/notes/randomness.html
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

for epoch in range(3):
    print(f"-----epoch {epoch} ------")
    for data in dataloader:
        print(data)
