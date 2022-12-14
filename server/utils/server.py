import os
from server.model.CNN import CNN
from server.model.GRU import GRU
from server.model.LSTM import LSTM
from server.utils.data_manager import load_test, MyDataset
from server.utils.calculate import score, score_plot
from server.network.network import network
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
        self.device = 'cpu'
        self.model_type = 0  # 0:GRU, 1:LSTM 2:CNN
        self.optimizer_type = 1  # 0:sgd, 1:adam
        self.num_workers = 0
        self.learning_rate = 0.001
        self.weight_decay = 5e-4
        self.momentum = 0.9
        self.fed_algorithm = 'fedavg'  # fedavg/fedavg+centerloss/fedprox/fedprox+centerloss/moon/moon+centerloss
        self.fedprox_mu = 0.01  # 0.12  # 0.3
        self.local_epoch = 1  # IoT_FD epoch for each communication epoch
        self.global_epoch = 2
        self.num_devices = 5
        self.data = None
        self.label = None
        self.test_loader = None
        self.importance_dict = {}
        self.feature_size = 0
        self.evaluation_client = False
        self.score_threshold = 0.01
        self.model = None

        # network
        self.device_epoch = 1  # IoT_FD epoch for each communication epoch
        self.federated_epoch = 4
        self.address = '10.201.45.56'
        self.port = 8080
        self.network = None

    def load_client(self):
        self.network = network(self.address, self.port)
        self.network.load_client()

    def init_client(self):
        send_message = {
            "batch_size": self.batch_size,
            "attack_dic": self.attack_dict,
            "model_type": self.model_type,
            "fed_algorithm": self.fed_algorithm,
            "local_epoch": self.local_epoch,
            "global_epoch": self.global_epoch
        }
        self.network.socket_send_init_client(send_message)

    def load_dataset(self):
        self.data, self.label = load_test(self.attack_dict)
        header = self.data.columns
        for key in header:
            self.importance_dict.update({key: 0})

    def feature_reduction(self):
        result = self.network.socket_receive_feature_selection()
        for importance in result:
            for key, value in importance.items():
                self.importance_dict[key] += value

        drop_columns = []
        self.importance_dict = dict(sorted(self.importance_dict.items(), key=lambda x: x[1], reverse=False))
        for key, value in self.importance_dict.items():
            if value < self.score_threshold:
                drop_columns.append(key)
        self.network.socket_send_drop_columns(drop_columns)

        self.data = self.data.drop(drop_columns, axis=1)
        self.feature_size = len(self.data.columns)
        test_set = MyDataset(self.data, self.label, len(self.attack_dict))
        self.test_loader = DataLoader(dataset=test_set,
                                      batch_size=self.batch_size,
                                      shuffle=True,
                                      num_workers=0,
                                      drop_last=True)
        logger.info("left columns:{}".format(str(self.feature_size)))

    def init_client_model(self):
        if self.model_type == 0:
            self.model = GRU(input_size=self.feature_size,
                             hidden_layer_size=256,
                             num_layers=2,
                             output_size=len(self.attack_dict),
                             dropout=0)
        elif self.model_type == 1:
            self.model = LSTM(input_size=self.feature_size,
                              output_size=len(self.attack_dict),
                              hidden_layer_size=256,
                              num_layers=2,
                              dropout=0)
        else:
            self.model = CNN(input_size=self.feature_size,
                             output_size=len(self.attack_dict))
        model_state_dict = {"model": self.model.state_dict()}
        torch.save(model_state_dict, "./snapshot/global.pth")
        self.network.socket_send_file()

    def aggregation(self):
        s = score_plot(self.network.target)
        for global_epoch in range(0, self.global_epoch):
            self.network.socket_receive_file()
            aggregation_parameter = {}
            model_parameter_list = []
            model_weight_list = []
            for key, value in self.network.target.items():
                filename = './snapshot/' + key.replace(":", ",") + '.pth'
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

            logger.info("------------------global model-------------------")
            model_path = './snapshot/global.pth'
            model_state_dict = {"model": aggregation_parameter}
            torch.save(model_state_dict, model_path)
            self.model.load_state_dict(model_state_dict["model"])
            accuracy, precision, recall, f1_score = self.evaluation()
            s.update("global", global_epoch, accuracy, precision, recall, f1_score)
            for key, value in self.network.target.items():
                logger.info("------------------" + key + " model-------------------")
                model_state_dict = torch.load('./snapshot/' + key.replace(":", ",") + '.pth')
                self.model.load_state_dict(model_state_dict["model"])
                accuracy, precision, recall, f1_score = self.evaluation()
                s.update(key, global_epoch, accuracy, precision, recall, f1_score)
            self.network.socket_send_file()

        self.network.close()
        s.plot()

    def evaluation(self):
        start = time.time()
        self.model.eval()
        s = score(self.attack_dict)
        device = torch.device(self.device)
        with torch.no_grad():
            for batch, (data, label) in enumerate(self.test_loader):
                data = data.to(device)
                label = label.to(device)
                _, test_output = self.model(data)
                predict_label = torch.argmax(test_output, dim=1)
                true_label = label.view(-1)
                s.update(predict_label, true_label)
        accuracy, precision, recall, f1_score = s.compute()
        end = time.time()
        logger.info("accuracy: {:.4f}%, precision:{:.4f}, recall:{:.4f}, test time:{:.4f}s"
                    .format(accuracy, precision, recall, end - start))
        del s
        return accuracy, precision, recall, f1_score
