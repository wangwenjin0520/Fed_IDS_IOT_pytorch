import numpy as np
import os
from client.utils.data_manager import load_train, MyDataset
from client.network.receive_manager import socket_service_init
from torch.utils.data import DataLoader
from sklearn.feature_selection import SelectKBest, chi2
import logging
import time
import torch
logger = logging.getLogger('global')


class client_info:
    def __init__(self, device_id=0, fed_algorithm='fedavg', model_type=0, federated_epoch=0):
        # init
        self.device_id = device_id
        self.batch_size = 16384
        self.attack_dict = {"mitm": 0, "scanning": 1, "dos": 2, "ddos": 3, "injection": 4, "password": 5,
                            "backdoor": 6, "ransomware": 7, "xss": 8, "Benign": 9}
        self.device = 'cuda'
        self.model_type = model_type  # 0:GRU, 1:LSTM 2:CNNGRU 3:CNN
        self.optimizer_type = 1  # 0:sgd, 1:adam
        self.num_workers = 0
        self.learning_rate = 0.001
        self.weight_decay = 5e-4
        self.momentum = 0.9
        self.fed_algorithm = fed_algorithm
        self.fedprox_mu = 0.01  # 0.12  # 0.3
        self.device_epoch = 1  # IoT_FD epoch for each communication epoch
        self.federated_epoch = federated_epoch
        self.data = None
        self.label = None
        self.train_loader = None
        self.dataset_header = None
        self.address = '192.168.255.1'
        self.port = 4000

    def init(self):
        socket_service_init(self.address, self.port)

    def load_dataset(self):
        self.data, self.label = load_train(self.attack_dict)

    def calculate_importance(self):
        importance_dict = {}
        header = list(self.data.columns)
        model_sk = SelectKBest(score_func=chi2, k=3)
        model_sk.fit(self.data, self.label)
        importance = model_sk.scores_
        importance = np.nan_to_num(importance)
        for key, value in zip(header, list(importance)):
            importance_dict.update({key: value})
        return importance_dict

    def feature_reduction(self, columns):
        self.data = self.data.drop(columns, axis=1)
        train_set = MyDataset(self.data, self.label, len(self.attack_dict))
        self.train_loader = DataLoader(dataset=train_set,
                                       batch_size=self.batch_size,
                                       shuffle=True,
                                       num_workers=0,
                                       drop_last=True)

    def train(self, model=None, tmp_model=None, criterions=None, optimizer=None):
        global_model_state = torch.load('./snapshot/after/global.pth')
        global_optimizer_state = torch.load('./snapshot/before/optimizer_device' + str(self.device_id) + '.pth')
        model.load_state_dict(global_model_state['model'])
        optimizer.load_state_dict(global_optimizer_state['optimizer'])
        device = torch.device(self.device)
        model.train()
        total_size = 0
        logger.info("-------------device {}---------------".format(str(self.device_id)))
        for epoch in range(self.device_epoch):
            # start training
            start = time.time()
            loss_last = 0.0
            batch = 0

            # start loop
            for batch, (data, label) in enumerate(self.train_loader):
                data = data.to(device)
                label = label.to(device)
                center_input, train_output = model(data)
                train_output.to(device)

                # fedprox
                if self.fed_algorithm == 'fedprox' or self.fed_algorithm == 'fedprox+centerloss':
                    original_model_state = torch.load('./snapshot/after/global.pth')
                    tmp_model.load(original_model_state['model'])
                    fedprox_loss = 0
                    proximal_term = 0.0
                    for w, w_t in zip(model.parameters(), tmp_model.parameters()):
                        proximal_term += (w - w_t).norm(2)
                        fedprox_loss += (self.fedprox_mu / 2) * proximal_term
                    cross_entropy_loss = criterions[0](train_output, label)
                    if self.fed_algorithm == 'fedprox':
                        loss = cross_entropy_loss + fedprox_loss
                    else:
                        center_loss = criterions[1](center_input, label)
                        loss = center_loss + cross_entropy_loss + fedprox_loss

                # fedmoon
                elif self.fed_algorithm == 'moon' or self.fed_algorithm == 'moon+centerloss':
                    prev_model_state = torch.load('./snapshot/before/model_device' + str(self.device_id) + '.pth')
                    tmp_model.load_state_dict(prev_model_state['model'])
                    tmp_model.to(torch.device(self.device))
                    prev_representation, _ = tmp_model(data)

                    original_model_state = torch.load('./snapshot/after/global.pth')
                    tmp_model.load_state_dict(original_model_state['model'])
                    tmp_model.to(torch.device(self.device))
                    global_representation, _ = tmp_model(data)

                    cos = torch.nn.CosineSimilarity(dim=-1)
                    criterion_tmp = torch.nn.CrossEntropyLoss()
                    current_prev = cos(center_input, prev_representation).reshape(-1, 1)
                    current_global = cos(center_input, global_representation).reshape(-1, 1)
                    logits = torch.cat((current_global, current_prev), dim=1)
                    fedmoon_label = torch.zeros(data.size(0)).cuda().long()
                    fedmoon_loss = criterion_tmp(logits, fedmoon_label).data
                    cross_entropy_loss = criterions[0](train_output, label)
                    if self.fed_algorithm == 'moon':
                        loss = cross_entropy_loss + 10 * fedmoon_loss
                    else:
                        center_loss = criterions[1](center_input, label)
                        loss = cross_entropy_loss + center_loss + 100 * fedmoon_loss

                else:
                    cross_entropy_loss = criterions[0](train_output, label)
                    if self.fed_algorithm == 'fedavg':
                        loss = cross_entropy_loss
                    else:
                        center_loss = criterions[1](center_input, label)
                        loss = cross_entropy_loss + center_loss

                optimizer.zero_grad()
                loss.backward(retain_graph=True)
                optimizer.step()
                loss_last = loss.data
            total_size = batch * self.batch_size
            # print progress
            end = time.time()
            logger.info('Epoch [{}/{}], Train Loss mean: {:.4f}, running time:{:.4f}s'
                        .format(epoch + 1, self.device_epoch, loss_last, end - start))

        os.remove('./snapshot/before/model_device' + str(self.device_id) + '.pth')
        os.remove('./snapshot/before/optimizer_device' + str(self.device_id) + '.pth')
        self.final_save(model, optimizer, total_size)
        logger.info('--------------------------------------------------------------')

    def final_save(self, model, optimizer, total_size):
        model_path = './snapshot/before/model_device' + str(self.device_id) + '.pth'
        optimizer_path = './snapshot/before/optimizer_device' + str(self.device_id) + '.pth'
        model_state_dic = {'model': model.state_dict(), 'total_size': total_size}
        optimizer_state_dic = {'optimizer': optimizer.state_dict()}
        torch.save(model_state_dic, model_path)
        torch.save(optimizer_state_dic, optimizer_path)
