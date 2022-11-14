import numpy as np
import os
from client.model.CNN import CNN
from client.model.GRU import GRU
from client.model.LSTM import LSTM
from client.loss.crossentropyloss import LabelSmoothCrossEntropyLoss
from client.loss.centerloss import CenterLoss
from client.utils.data_manager import load_train, MyDataset
from client.network.network import network
from torch.utils.data import DataLoader
from sklearn.feature_selection import SelectKBest, chi2
import logging
import time
import torch
import shutil
logger = logging.getLogger('global')


class client_info:
    def __init__(self, fed_algorithm='fedavg', model_type=0, federated_epoch=0):
        # init
        self.batch_size = 16384
        self.attack_dict = {"mitm": 0, "scanning": 1, "dos": 2, "ddos": 3, "injection": 4, "password": 5,
                            "backdoor": 6, "ransomware": 7, "xss": 8, "Benign": 9}
        self.device = 'cpu'
        self.model_type = model_type  # 0:GRU, 1:LSTM 2:CNNGRU 3:CNN
        self.optimizer_type = 1  # 0:sgd, 1:adam
        self.feature_size = 0
        self.num_workers = 0
        self.learning_rate = 0.001
        self.weight_decay = 5e-4
        self.momentum = 0.9
        self.fed_algorithm = fed_algorithm
        self.fedprox_mu = 0.01  # 0.12  # 0.3
        self.local_epoch = 1  # IoT_FD epoch for each communication epoch
        self.global_epoch = 400
        self.federated_epoch = federated_epoch
        self.data = None
        self.label = None
        self.total_size = 0
        self.train_loader = None
        self.dataset_header = None
        self.global_model = None
        self.tmp_model = None
        self.global_optimizer = None
        self.global_criterions = None
        self.address = '192.168.255.1'
        self.port = 4000
        self.target_address = '192.168.255.1'
        self.target_port = 8080
        self.network = None

    def init(self):
        self.network = network(self.address, self.port, self.target_address, self.target_port)
        self.network.build_connection()
        result = self.network.socket_receive_init_parameter()
        self.attack_dict = result["attack_dic"]
        self.model_type = result["model_type"]
        self.fed_algorithm = result["fed_algorithm"]
        self.local_epoch = result["local_epoch"]
        self.global_epoch = result["global_epoch"]

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
            importance_dict.update({key: round(value, 2)})
        self.network.socket_send_feature_importance(importance_dict)

    def feature_reduction(self):
        result = self.network.socket_receive_init_usecolumns()
        self.data = self.data.drop(result, axis=1)
        self.total_size = len(self.label)
        self.feature_size = len(self.data.columns)
        train_set = MyDataset(self.data, self.label, len(self.attack_dict))
        self.train_loader = DataLoader(dataset=train_set,
                                       batch_size=self.batch_size,
                                       shuffle=True,
                                       num_workers=0,
                                       drop_last=True)

    def criterion_init(self):
        # model
        if self.model_type == 0:
            model = GRU(input_size=self.feature_size,
                        hidden_layer_size=256,
                        num_layers=2,
                        output_size=len(self.attack_dict),
                        dropout=0)
        elif self.model_type == 1:
            model = LSTM(input_size=self.feature_size,
                         output_size=len(self.attack_dict),
                         hidden_layer_size=256,
                         num_layers=2,
                         dropout=0)
        else:
            model = CNN(input_size=self.feature_size,
                        output_size=len(self.attack_dict))

        # optimizers
        optimizer = torch.optim.Adam(model.parameters(),
                                     lr=self.learning_rate,
                                     weight_decay=self.weight_decay)

        criterions = [LabelSmoothCrossEntropyLoss(device=self.device),
                      CenterLoss(num_classes=len(self.attack_dict),
                                 feat_dim=256,
                                 device=self.device)]

        return model.to(torch.device(self.device)), optimizer, criterions

    def init_model(self):
        self.global_model, self.global_optimizer, self.global_criterions = self.criterion_init()
        if self.fed_algorithm != 'fedavg' and self.fed_algorithm != 'fedavg+centerloss':
            self.tmp_model, _, _ = self.criterion_init()
        self.network.socket_receive_file()
        shutil.copyfile("./snapshot/global.pth", "./snapshot/epoch0.pth")

    def train(self):
        for global_epoch in range(0, self.global_epoch):
            global_model_state = torch.load('./snapshot/global.pth')
            self.global_model.load_state_dict(global_model_state['model'])
            self.global_model.train()
            for local_epoch in range(self.local_epoch):
                # start training
                start = time.time()
                loss_last = 0.0

                # start loop
                for batch, (data, label) in enumerate(self.train_loader):
                    data = data.to(self.device)
                    label = label.to(self.device)
                    center_input, train_output = self.global_model(data)
                    train_output.to(self.device)

                    # fedprox
                    if self.fed_algorithm == 'fedprox' or self.fed_algorithm == 'fedprox+centerloss':
                        original_model_state = torch.load('./snapshot/epoch'+str(global_epoch)+'.pth')
                        self.tmp_model.load(original_model_state['model'])
                        fedprox_loss = 0
                        proximal_term = 0.0
                        for w, w_t in zip(self.global_model.parameters(), self.tmp_model.parameters()):
                            proximal_term += (w - w_t).norm(2)
                            fedprox_loss += (self.fedprox_mu / 2) * proximal_term
                        cross_entropy_loss = self.global_criterions[0](train_output, label)
                        if self.fed_algorithm == 'fedprox':
                            loss = cross_entropy_loss + fedprox_loss
                        else:
                            center_loss = self.global_criterions[1](center_input, label)
                            loss = center_loss + cross_entropy_loss + fedprox_loss

                    # fedmoon
                    elif self.fed_algorithm == 'moon' or self.fed_algorithm == 'moon+centerloss':
                        prev_model_state = torch.load('./snapshot/before/epoch'+str(global_epoch)+'.pth')
                        self.tmp_model.load_state_dict(prev_model_state['model'])
                        self.tmp_model.to(torch.device(self.device))
                        prev_representation, _ = self.tmp_model(data)

                        original_model_state = torch.load('./snapshot/after/global.pth')
                        self.tmp_model.load_state_dict(original_model_state['model'])
                        self.tmp_model.to(torch.device(self.device))
                        global_representation, _ = self.tmp_model(data)

                        cos = torch.nn.CosineSimilarity(dim=-1)
                        criterion_tmp = torch.nn.CrossEntropyLoss()
                        current_prev = cos(center_input, prev_representation).reshape(-1, 1)
                        current_global = cos(center_input, global_representation).reshape(-1, 1)
                        logits = torch.cat((current_global, current_prev), dim=1)
                        fedmoon_label = torch.zeros(data.size(0)).cuda().long()
                        fedmoon_loss = criterion_tmp(logits, fedmoon_label).data
                        cross_entropy_loss = self.global_criterions[0](train_output, label)
                        if self.fed_algorithm == 'moon':
                            loss = cross_entropy_loss + 10 * fedmoon_loss
                        else:
                            center_loss = self.global_criterions[1](center_input, label)
                            loss = cross_entropy_loss + center_loss + 100 * fedmoon_loss

                    else:
                        cross_entropy_loss = self.global_criterions[0](train_output, label)
                        if self.fed_algorithm == 'fedavg':
                            loss = cross_entropy_loss
                        else:
                            center_loss = self.global_criterions[1](center_input, label)
                            loss = cross_entropy_loss + center_loss

                    self.global_optimizer.zero_grad()
                    loss.backward(retain_graph=True)
                    self.global_optimizer.step()
                    loss_last = loss.data
                # print progress
                end = time.time()
                logger.info('Epoch [{}/{}], Train Loss mean: {:.4f}, running time:{:.4f}s'
                            .format(global_epoch + 1, self.global_epoch, loss_last, end - start))

            model_state_dict = {"model": self.global_model.state_dict(), "total_size":self.total_size}
            torch.save(model_state_dict, "./snapshot/epoch"+str(global_epoch+1)+".pth")
            os.remove('./snapshot/epoch' + str(global_epoch) + '.pth')
            self.network.socket_send_file(global_epoch+1)
            self.network.socket_receive_file()
            logger.info('--------------------------------------------------------------')
        self.network.close()
