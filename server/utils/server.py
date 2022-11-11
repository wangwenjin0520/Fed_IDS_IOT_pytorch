import os

from server.utils.data_manager import load_test, MyDataset
from server.utils.calculate import score
from server.network.send_manager import socket_send_init_client
from torch.utils.data import DataLoader
import torch
import time
import logging
logger = logging.getLogger('global')


class server_info:
    def __init__(self):
        # init
        self.batch_size = 4096
        self.attack_dict = {"mitm": 0, "scanning": 1, "dos": 2, "ddos": 3, "injection": 4, "password": 5,
                            "backdoor": 6, "ransomware": 7, "xss": 8, "Benign": 9}
        self.device = 'cuda'
        self.model_type = 0  # 0:GRU, 1:LSTM 2:CNN
        self.optimizer_type = 1  # 0:sgd, 1:adam
        self.num_workers = 0
        self.learning_rate = 0.001
        self.weight_decay = 5e-4
        self.momentum = 0.9
        self.fed_algorithm = 'fedavg'  # fedavg/fedavg+centerloss/fedprox/fedprox+centerloss/moon/moon+centerloss
        self.fedprox_mu = 0.01  # 0.12  # 0.3
        self.local_epoch = 1  # IoT_FD epoch for each communication epoch
        self.global_epoch = 400
        self.num_devices = 5
        self.data = None
        self.label = None
        self.test_loader = None
        self.importance_dict = {}
        self.feature_size = 0
        self.evaluation_client = False
        self.score_threshold = 0.01

        # network
        self.device_epoch = 1  # IoT_FD epoch for each communication epoch
        self.federated_epoch = 4
        self.address = '192.168.255.1'
        self.port = 8080
        self.target = []

    def load_client(self):
        file = open('./utils/config.txt', 'r')
        lines = file.readlines()
        for (index, line) in enumerate(lines):
            client = line.split(",")
            self.target.append({"address": client[0], "port": int(client[1]), "id": index})

    def init_client(self):
        send_message = {
            "attack_dic": self.attack_dict,
            "model_type": self.model_type,
            "fed_algorithm": self.fed_algorithm,
            "local_epoch": self.local_epoch,
            "global_epoch": self.global_epoch
        }
        for client in self.target:
            socket_send_init_client(send_message, client["address"], client["port"])

    def load_dataset(self):
        self.data, self.label = load_test(self.attack_dict)
        header = self.data.columns
        for key in header:
            self.importance_dict.update({key: 0})

    def importance_append(self, importance):
        for key, value in importance.items():
            self.importance_dict[key] += value

    def importance_calculation(self):
        drop_columns = []
        self.importance_dict = dict(sorted(self.importance_dict.items(), key=lambda x: x[1], reverse=False))
        for key, value in self.importance_dict.items():
            if value < self.score_threshold:
                drop_columns.append(key)
        return drop_columns

    def feature_reduction(self, columns):
        self.data = self.data.drop(columns, axis=1)
        self.feature_size = len(self.data.columns)
        test_set = MyDataset(self.data, self.label, len(self.attack_dict))
        self.test_loader = DataLoader(dataset=test_set,
                                      batch_size=self.batch_size,
                                      shuffle=True,
                                      num_workers=0,
                                      drop_last=True)
        logger.info("left columns:{}".format(str(self.feature_size)))

    def aggregation(self):
        os.remove('./snapshot/after/global.pth')
        aggregation_parameter = {}
        model_parameter_list = []
        model_weight_list = []
        for i in range(self.num_devices):
            filename = './snapshot/before/model_device' + str(i) + '.pth'
            state_dict = torch.load(filename)
            model_parameter = {}
            for key, var in state_dict['model'].items():
                model_parameter.update({key: var})
            model_parameter_list.append(model_parameter)
            model_weight_list.append(state_dict["total_size"])

        weight_sum = sum(model_weight_list)
        for i in range(len(model_weight_list)):
            if not aggregation_parameter:
                for key, var in model_parameter_list[i].items():
                    aggregation_parameter.update({key: var * model_weight_list[i] / weight_sum})
            else:
                for key, var in model_parameter_list[i].items():
                    aggregation_parameter[key] += var * model_weight_list[i] / weight_sum
        model_path = './snapshot/after/global.pth'
        model_state_dict = {"model": aggregation_parameter}
        torch.save(model_state_dict, model_path)

    def evaluation(self, model):
        start = time.time()
        model.eval()
        s = score(self.attack_dict)
        device = torch.device(self.device)
        with torch.no_grad():
            for batch, (data, label) in enumerate(self.test_loader):
                data = data.to(device)
                label = label.to(device)
                _, test_output = model(data)
                predict_label = torch.argmax(test_output, dim=1)
                true_label = label.view(-1)
                s.update(predict_label, true_label)
        accuracy, precision, recall, f1_score = s.compute()
        end = time.time()
        logger.info("accuracy: {:.4f}%, precision:{:.4f}, recall:{:.4f}, test time:{:.4f}s"
                    .format(accuracy, precision, recall, end - start))
        del s
        return accuracy, precision, recall, f1_score
