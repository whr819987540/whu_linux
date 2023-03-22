import torch


class ServeParty(object):
    def __init__(self, model, loss_func, optimizer, n_iter=1):
        super(ServeParty, self).__init__()
        self.model = model
        self.loss_func = loss_func
        self.optimizer = optimizer
        self.n_iter = n_iter
        
        self.h_dim_list = []
        self.parties_grad_list = []
        self.y = None
        self.batch_size = None
        self.h_input = None

    def set_batch(self, y):
        self.y = y
        self.batch_size = y.shape[0]

    def pull_parties_h(self, h_list):
        self.h_dim_list = [h.shape[1] for h in h_list]
        h_input = None
        for h in h_list:
            if h_input is None:
                h_input = h
            else:
                h_input = torch.cat([h_input, h], 1)
        h_input = h_input.detach()
        h_input.requires_grad = True
        self.h_input = h_input

    def compute_parties_grad(self):
        output = self.model(self.h_input)
        loss = self.loss_func(output, self.y)
        loss.backward()
        parties_grad = self.h_input.grad

        self.parties_grad_list = []
        start = 0
        for dim in self.h_dim_list:
            self.parties_grad_list.append(parties_grad[:, start:start+dim])
            start += dim
    
    def send_parties_grad(self):
        return self.parties_grad_list

    def local_update(self):
        for para in self.model.parameters():
            para.grad /= self.batch_size
        self.optimizer.step()
        self.optimizer.zero_grad()
    
    def local_iterations(self):
        for i in range(self.n_iter-1):
            self.compute_parties_grad()
            self.local_update()

    def predict(self, h_list, y):
        batch_size = y.shape[0]
        self.pull_parties_h(h_list)
        self.model.eval()
        with torch.no_grad():
            output = self.model(self.h_input)
            loss = self.loss_func(output, y)/batch_size
            pred = output.argmax(dim=1, keepdim=True)
            correct = pred.eq(y.view_as(pred)).sum().item()
            accuracy = correct/batch_size

        return loss, correct, accuracy


class ClientParty(object):
    def __init__(self, model, optimizer, n_iter=1):
        super(ClientParty, self).__init__()
        self.model = model
        self.optimizer = optimizer
        self.n_iter = n_iter

        self.x = None
        self.h = None
        self.partial_grad = None
        self.batch_size = None

    def set_batch(self, x):
        self.x = x
        self.batch_size = x.shape[0]
    
    def compute_h(self):
        self.h = self.model(self.x)

    def send_h(self):
        return self.h

    def pull_grad(self, grad):
        self.partial_grad = grad

    def local_update(self):
        self.h.backward(self.partial_grad)
        for para in self.model.parameters():
            para.grad /= self.batch_size
        self.optimizer.step()
        self.optimizer.zero_grad()

    def local_iterations(self):
        for i in range(self.n_iter-1):
            self.compute_h()
            self.local_update()

    def predict(self, x):
        predict_h = self.model(x)
        return predict_h
