import time
import torch
import logging
import os
from client.utils.client import client_info
from client.model.CNN import CNN
from client.model.GRU import GRU
from client.model.LSTM import LSTM
from client.utils.log_helper import init_log, add_file_handler
from client.loss.crossentropyloss import LabelSmoothCrossEntropyLoss
from client.loss.centerloss import CenterLoss
from client.utils.memory import Monitor

logger = logging.getLogger('global')


def criterion_init(model_type=0, device='cuda', feature_size=0, num_classes=0, learning_rate=0.01, weight_decay=0.1):
    # model
    if model_type == 0:
        model = GRU(input_size=feature_size,
                    hidden_layer_size=256,
                    num_layers=2,
                    output_size=num_classes,
                    dropout=0)
    elif model_type == 1:
        model = LSTM(input_size=feature_size,
                     output_size=num_classes,
                     hidden_layer_size=256,
                     num_layers=2,
                     dropout=0)
    else:
        model = CNN(input_size=feature_size,
                    output_size=num_classes)

    # optimizers
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=learning_rate,
                                 weight_decay=weight_decay)

    criterions = [LabelSmoothCrossEntropyLoss(device=device),
                  CenterLoss(num_classes=num_classes,
                             feat_dim=256,
                             device=device)]

    return model.to(torch.device(device)), optimizer, criterions


if __name__ == '__main__':
    #a = Monitor(1)
    #a.start()

    # init log
    init_log('global', logging.INFO)
    add_file_handler('global', './logs/logs.txt', logging.INFO)
    logger.info("log init done")

    # init server and client
    client = client_info()
    client.init()

    # init dataset
    start_time = time.time()
    client.load_dataset()
    logger.info("dataset build done. time:{:.4f}".format(time.time() - start_time))


    # calculate importance
    start_time = time.time()
    client.calculate_importance()
    client.feature_reduction()
    logger.info("importance finished, time:{:.4f}".format(time.time() - start_time))

    # init model
    client.init_model()

    # IoT_FD
    logger.info("start training")
    client.train()
