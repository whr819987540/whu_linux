import torch
import torchvision
from torch.utils.data import DataLoader
dataset_path = '/home/whr-pc-ubuntu/code/dataset'
batch_size = 10000
seed = 10

trans = torchvision.transforms.Compose(
    [
        torchvision.transforms.ToTensor(),
    ]
)
train_data = torchvision.datasets.MNIST(
    root=dataset_path,
    train=True,
    transform=trans,
    download=True
)

test_data = torchvision.datasets.MNIST(
    root=dataset_path,
    train=False,
    transform=trans,
    download=True
)

torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

train_dataloader = DataLoader(
    train_data,
    batch_size=batch_size,
    shuffle=True
)
test_dataloader = DataLoader(
    test_data,
    batch_size=batch_size,
    shuffle=True
)

for epoch in range(3):
    print(f"------ epoch {epoch} ------")
    for data in train_dataloader:
        print(data)
