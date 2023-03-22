import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from models.a9a_model import ClientNet, ServerNet
from partymodel import ServeParty, ClientParty
from learner import VFLlearner

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class A9ADataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels
        self.size = data.shape[0]

    def __getitem__(self, index):
        return self.data[index], self.labels[index]

    def __len__(self):
        return self.size


def load_data(path):
    data = []
    labels = []
    file = open(path, 'r')
    file_data = file.readlines()
    for row in file_data:
        tmp_list = row.split(' ')
        labels.append(1 if int(tmp_list[0]) == 1 else 0)
        one_row = [0]*123
        for val in tmp_list[1:-1]:
            one_row[int(val.split(':')[0])-1] = 1
        data.append(one_row)
    
    data = torch.Tensor(data)
    labels = torch.Tensor(labels).long()
    return data, labels


def run_experiment():
    data_dir = 'data/a9a'
    output_dir = "summary_pic/a9a/"
    parties_num = 4
    n_local = [5, 5, 5, 5]
    bound = [5, 5, 5]
    delay_factor = [0, 0, 0]
    batch_size = 256
    epochs = 2

    div = [10, 33]

    train_data, train_labels = load_data(data_dir+'/train.txt')
    test_data, test_labels = load_data(data_dir+'/test.txt')

    data_loader_list = []
    party_list = []
    div = [0]+div+[train_data.shape[1]]
    for i in range(parties_num):
        if i == 0:
            train_dataset = A9ADataset(train_data, train_labels)
            test_dataset = A9ADataset(test_data, test_labels)

            model = ServerNet(3).to(device)
            loss_func = nn.CrossEntropyLoss().to(device)
            optimizer = optim.Adam(model.parameters())
            party = ServeParty(model=model, loss_func=loss_func, optimizer=optimizer, n_iter=n_local[i])
        else:
            train_dataset = A9ADataset(train_data[:, div[i-1]:div[i]], train_labels)
            test_dataset = A9ADataset(test_data[:, div[i-1]:div[i]], test_labels)

            model = ClientNet(div[i]-div[i-1]).to(device)
            optimizer = optim.Adam(model.parameters())
            party = ClientParty(model=model, optimizer=optimizer, n_iter=n_local[i])
        
        train_loader = DataLoader(train_dataset, batch_size)
        test_loader = DataLoader(test_dataset, batch_size)
        data_loader_list.append([train_loader, test_loader])
        party_list.append(party)

    print("################################ Train Federated Models ############################")

    vfl_learner = VFLlearner(party_list, data_loader_list, epochs, bound, delay_factor, output_dir)
    vfl_learner.start_learning()


if __name__ == '__main__':
    run_experiment()
