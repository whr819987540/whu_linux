import torch
import time
import random
import math
from threading import Thread
from queue import Queue
from torch.utils.tensorboard import SummaryWriter

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class VFLlearner(object):
    def __init__(self, party_list, data_loader_list, epochs, bound, delay_factor, output_dir):
        super(VFLlearner, self).__init__()
        self.server_party = party_list[0]
        self.server_data_loader = data_loader_list[0]
        self.client_party_list = party_list[1:]
        self.client_data_loader_list = data_loader_list[1:]
        self.epochs = epochs
        self.bound = bound
        self.delay_factor = delay_factor
        self.output_dir = output_dir

        self.server_party_thread = None
        self.client_party_thread_list = []
        self.parties_num = len(party_list)

    def start_learning(self):
        parties_h_queue_list = []
        parties_predict_h_queue_list = []
        parties_grad_queue_list = []
        for i in range(self.parties_num-1):
            parties_h_queue_list.append(Queue())
            parties_predict_h_queue_list.append(Queue())
            parties_grad_queue_list.append(Queue())

        server_data = {
            'train_loader': self.server_data_loader[0],
            'test_loader': self.server_data_loader[1],
            'party': self.server_party,
            'epochs': self.epochs,
            'output_dir': self.output_dir,
            'parties_h_queue_list': parties_h_queue_list,
            'parties_predict_h_queue_list': parties_predict_h_queue_list,
            'parties_grad_queue_list': parties_grad_queue_list
            }
        self.server_party_thread = ServerPartyThread(server_data, 0)

        for id, party in enumerate(self.client_party_list):
            client_data = {
                'train_loader': self.client_data_loader_list[id][0],
                'test_loader': self.client_data_loader_list[id][1],
                'party': party,
                'epochs': self.epochs,
                'bound': self.bound[id],
                'delay_factor': self.delay_factor[id],
                'h_queue': parties_h_queue_list[id],
                'predict_h_queue': parties_predict_h_queue_list[id],
                'grad_queue': parties_grad_queue_list[id],
                }
            self.client_party_thread_list.append(ClientPartyThread(client_data, id+1))
        
        self.server_party_thread.start()
        for thread in self.client_party_thread_list:
            thread.start()


class ServerPartyThread(Thread):
    def __init__(self, data, thread_id):
        super(ServerPartyThread, self).__init__()
        self.data = data
        self.thread_id = thread_id

    def run(self):
        train_loader = self.data['train_loader']
        test_loader = self.data['test_loader']
        party = self.data['party']
        epochs = self.data['epochs']
        output_dir = self.data['output_dir']
        parties_h_queue_list = self.data['parties_h_queue_list']
        parties_predict_h_queue_list = self.data['parties_predict_h_queue_list']
        parties_grad_queue_list = self.data['parties_grad_queue_list']
        
        recording_period = math.ceil(len(train_loader) / len(test_loader))
        global_step = 0
        running_time = 0

        writer = SummaryWriter(log_dir=output_dir+time.strftime("%Y%m%d-%H%M"))

        print(f'thread{self.thread_id}: server start with batches={len(train_loader)}')

        for ep in range(epochs):
            print(f'thread{self.thread_id}: server start epoch {ep}')
            it = iter(test_loader)
            for batch_idx, (_, target) in enumerate(train_loader):
                party.model.train()
                start_time = time.time()

                target = target.to(device)
                party.set_batch(target)
                print(f'thread{self.thread_id}: server set batch {batch_idx}\n')
                
                h_list = []
                for q in parties_h_queue_list:
                    h_list.append(q.get())
                
                party.pull_parties_h(h_list)
                party.compute_parties_grad()
                parties_grad_list = party.send_parties_grad()

                for id, q in enumerate(parties_grad_queue_list):
                    q.put(parties_grad_list[id])
                party.local_update()
                party.local_iterations()

                end_time = time.time()
                spend_time = end_time - start_time
                running_time += spend_time

                global_step += 1

                if global_step % recording_period == 0:
                    predict_h_list = []
                    for q in parties_predict_h_queue_list:
                        predict_h_list.append(q.get())
                    predict_y = next(it)[1].to(device)
                    loss, correct, accuracy = party.predict(predict_h_list, predict_y)
                    writer.add_scalar("loss", loss.detach(), global_step)
                    writer.add_scalar("accuracy&step", accuracy, global_step)
                    writer.add_scalar("accuracy&time", accuracy, running_time*1000)
                    writer.add_scalar("running_time", running_time, global_step)
                    print(f'thread{self.thread_id}: server figure out loss={loss} correct={correct} accuracy={accuracy} spend_time={spend_time}\n')
        
        writer.close()


class ClientPartyThread(Thread):
    def __init__(self, data, thread_id):
        super(ClientPartyThread, self).__init__()
        self.data = data
        self.thread_id = thread_id

    def run(self):
        train_loader = self.data['train_loader']
        test_loader = self.data['test_loader']
        party = self.data['party']
        epochs = self.data['epochs']
        bound = self.data['bound']
        h_queue = self.data['h_queue']
        predict_h_queue = self.data['predict_h_queue']
        grad_queue = self.data['grad_queue']
        delay_factor = self.data['delay_factor']

        train_batches = len(train_loader)
        test_batches = len(test_loader)
        recording_period = math.ceil(train_batches / test_batches)
        global_step = 0
        waiting_grad_num = 0

        batch_cache = Queue()

        send_buffer = Queue()
        pull_buffer = Queue()
        send_thread = Thread(target=self.process_communicate, daemon=True, args=(send_buffer, h_queue, delay_factor, self.thread_id, 'send'))
        pull_thread = Thread(target=self.process_communicate, daemon=True, args=(grad_queue, pull_buffer, delay_factor, self.thread_id, 'pull'))
        send_thread.start()
        pull_thread.start()

        print(f'thread{self.thread_id}: client start with batches={len(train_loader)}')

        it = iter(test_loader)
        for ep in range(epochs):
            print(f'thread{self.thread_id}: client start epoch {ep}\n')
            for batch_idx, (data, _) in enumerate(train_loader):
                party.model.train()

                data = data.to(device)
                party.set_batch(data)
                batch_cache.put([batch_idx, data])
                print(f'thread{self.thread_id}: client set batch {batch_idx}\n')

                party.compute_h()
                send_buffer.put(party.send_h())
                waiting_grad_num += 1

                while waiting_grad_num > bound or not pull_buffer.empty() or (ep == epochs-1 and batch_idx == train_batches-1 and waiting_grad_num > 0):
                    grad = pull_buffer.get()
                    cache_idx, batch_x = batch_cache.get()
                    party.set_batch(batch_x)
                    print(f'thread{self.thread_id}: client local update with batch {cache_idx}\n')
                    party.pull_grad(grad)
                    # party.compute_h()  # 这个好像不需要
                    party.local_update()
                    party.local_iterations()

                    waiting_grad_num -= 1
                    global_step += 1

                    if cache_idx == 0:
                        it = iter(test_loader)

                    if global_step % recording_period == 0:
                        predict_x = next(it)[0].to(device)
                        predict_h_queue.put(party.predict(predict_x))
    
    def process_local_update(self, party, grad_queue, batch_cache):
        grad = grad_queue.get()
        idx, batch_x = batch_cache.get()
        party.set_batch(batch_x)
        print(f'thread{self.thread_id}: client local update with batch {idx}\n')
        party.pull_grad(grad)
        party.compute_h()
        party.local_update()
        party.local_iterations()

    def process_communicate(self, src, dst, delay_factor, client_thread_id, task_name):
        print(f'{task_name} thread of client{client_thread_id} start\n')
        while(True):
            data = src.get()
            sleep_time = random.random()*delay_factor
            print(f'{task_name} thread of client{client_thread_id} sleep {sleep_time}\n')
            time.sleep(sleep_time)
            dst.put(data)
        print(f'{task_name} thread of client{client_thread_id} end\n')
