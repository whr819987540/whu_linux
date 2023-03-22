import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from models.cifar_model import ClientNet, ServerNet
from partymodel import ServeParty, ClientParty
from learner import VFLlearner

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def run_experiment():
    data_dir = 'data/cifar10'
    output_dir = "summary_pic/cifar10/"
    n_local = [5, 5, 5]
    bound = [0, 0]
    delay_factor = [0, 0]
    batch_size = 256
    epochs = 10

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    div = 20

    server_train_data = datasets.CIFAR10(data_dir, train=True, download=True, transform=transform)
    server_test_data = datasets.CIFAR10(data_dir, train=False, transform=transform)

    client1_train_dataset = datasets.CIFAR10(data_dir, train=True, download=True, transform=transform)
    client1_test_dataset = datasets.CIFAR10(data_dir, train=False, transform=transform)
    client1_train_dataset.data = client1_train_dataset.data[:, :div]
    client1_test_dataset.data = client1_test_dataset.data[:, :div]

    client2_train_dataset = datasets.CIFAR10(data_dir, train=True, download=True, transform=transform)
    client2_test_dataset = datasets.CIFAR10(data_dir, train=False, transform=transform)
    client2_train_dataset.data = client2_train_dataset.data[:, div:]
    client2_test_dataset.data = client2_test_dataset.data[:, div:]

    server_train_loader = DataLoader(server_train_data, batch_size=batch_size)
    server_test_loader = DataLoader(server_test_data, batch_size=batch_size)

    client1_train_loader = DataLoader(client1_train_dataset, batch_size=batch_size)
    client1_test_loader = DataLoader(client1_test_dataset, batch_size=batch_size)

    client2_train_loader = DataLoader(client2_train_dataset, batch_size=batch_size)
    client2_test_loader = DataLoader(client2_test_dataset, batch_size=batch_size)
    
    data_loader_list = [[server_train_loader, server_test_loader], [client1_train_loader, client1_test_loader],
                        [client2_train_loader, client2_test_loader]]

    server_model = ServerNet(2).to(device)
    server_loss_func = nn.CrossEntropyLoss()
    server_optimizer = optim.Adam(server_model.parameters())

    client1_model = ClientNet(3*32*20).to(device)
    client1_optimizer = optim.Adam(client1_model.parameters())

    client2_model = ClientNet(3*32*12).to(device)
    client2_optimizer = optim.Adam(client2_model.parameters())

    server_party = ServeParty(model=server_model, loss_func=server_loss_func, optimizer=server_optimizer, n_iter=n_local[0])
    client1_party = ClientParty(model=client1_model, optimizer=client1_optimizer, n_iter=n_local[1])
    client2_party = ClientParty(model=client2_model, optimizer=client2_optimizer, n_iter=n_local[2])

    party_list = [server_party, client1_party, client2_party]

    print("################################ Train Federated Models ############################")

    vfl_learner = VFLlearner(party_list, data_loader_list, epochs, bound, delay_factor, output_dir)
    vfl_learner.start_learning()


if __name__ == '__main__':
    run_experiment()
