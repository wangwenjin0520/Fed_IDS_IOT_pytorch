import time
import torch
import logging
import os
import shutil
from torch.nn import MultiMarginLoss
from server.utils.data_manager import build_data_loader
from server.model.GRU import GRU
from server.model.LSTM import LSTM
from server.utils.log_helper import init_log, add_file_handler
from server.utils.serialization import save_checkpoint
from server.utils.device_info import mydevice
from server.loss.crossentropyloss import LabelSmoothCrossEntropyLoss
from server.loss.centerloss import CenterLoss
from server.network.send_manager import socket_client
from server.network.receive_manager import socket_service_file, socket_service_init

logger = logging.getLogger('global')


def del_file(path_data):
    for i in os.listdir(path_data):
        file_data = path_data + "/" + i
        if os.path.isfile(file_data):
            os.remove(file_data)
        else:
            del_file(file_data)


def calculate_loss(criterion, predict_label, true_label, centerloss_input):
    loss = 0.0
    for index, loss_function in enumerate(criterion):
        if index != 1:
            loss += loss_function(predict_label, true_label)
        else:
            loss += loss_function(centerloss_input, true_label)
    return loss


def train(save_epoch, train_set, model, criterions, optimizer, device):
    model.train()
    for epoch in range(mydevice.device_epoch):
        # start training
        start = time.time()
        loss_last = 0.0
        # start loop
        for batch, (data, label) in enumerate(train_set):
            data = data.to(device)
            label = label.to(device)
            optimizer.zero_grad()
            center_input, train_output, hidden_cell_n = model(data)
            train_output.to(device)
            loss = calculate_loss(criterions, train_output, label, center_input)
            loss.backward(retain_graph=True)
            optimizer.step()
            if (batch + 1) % mydevice.print_frequency == 0:
                logger.info('Epoch [{}/{}], Batch [{}], Loss [{:.4f}]'
                            .format(epoch + 1, mydevice.device_epoch, batch + 1, loss.data))
            loss_last = loss.data
        # print progress
        end = time.time()
        logger.info('Epoch [{}/{}], Train Loss mean: {:.4f}, running time:{:.0f}s'
                    .format(epoch + 1, mydevice.device_epoch, loss_last, end - start))

    path_dir = './snapshot/send/client' + str(mydevice.device_id)
    if not os.path.exists(path_dir):
        os.mkdir(path_dir)
    save_checkpoint(path_dir + '/epoch' + str(save_epoch) + '.pth', model)
    logger.info('--------------------------------------------------------------')


def log_init():
    # log
    init_log('global', logging.INFO)
    add_file_handler('global', './logs/logs_' + str(mydevice.device_id) + '.txt', logging.INFO)
    logger.info("log init done")


def datasets_init():
    # datasets_seperation
    start_build_datasets = time.time()
    train_set, feature_size = build_data_loader()
    logger.info("dataset build done. time:{:.4f}".format(time.time() - start_build_datasets))
    return train_set, feature_size


def criterion_init(feature_size, num_classes):
    device = torch.device(mydevice.device)
    # model
    if mydevice.model_type == 0:
        model = GRU(input_size=feature_size,
                    hidden_layer_size=256,
                    num_layers=2,
                    output_size=num_classes,
                    dropout=0)
    else:
        model = LSTM(input_size=feature_size,
                     output_size=num_classes,
                     hidden_layer_size=256,
                     num_layers=2,
                     dropout=0)
    model.to(device)

    # loss
    criterions = [LabelSmoothCrossEntropyLoss(device=mydevice.device),
                  CenterLoss(num_classes=num_classes,
                             feat_dim=256,
                             device=mydevice.device)]

    # optimizers
    if mydevice.optimizer_type == "sgd":
        optimizer = torch.optim.SGD(model.parameters(),
                                    lr=mydevice.learning_rate,
                                    momentum=mydevice.momentum,
                                    weight_decay=mydevice.weight_decay)
    else:
        optimizer = torch.optim.Adam(model.parameters(),
                                     lr=mydevice.learning_rate,
                                     weight_decay=mydevice.weight_decay)

    return model, optimizer, criterions, device


def train_baseline():
    log_init()
    socket_service_init()
    train_set, feature_size = datasets_init()
    num_classes = len(mydevice.attack_dic)
    model, optimizer, criterions, device = criterion_init(feature_size=feature_size,
                                                          num_classes=num_classes)
    for federated_epoch in range(mydevice.federated_epoch):
        train(save_epoch=federated_epoch,
              train_set=train_set,
              model=model,
              criterions=criterions,
              optimizer=optimizer,
              device=device)
        socket_client('./snapshot/send/client' + str(mydevice.device_id) + '/epoch' + str(federated_epoch) + '.pth')
        socket_service_file(federated_epoch)
        stat_dict = torch.load(
            './snapshot/receive/client' + str(mydevice.device_id) + '/epoch' + str(federated_epoch) + '.pth',
            map_location=torch.device('cpu'))
        model.load_state_dict(stat_dict["model"])

    # copy the file to final model
    path_dir = './snapshot/final/client' + str(mydevice.device_id)
    if not os.path.exists(path_dir):
        os.mkdir(path_dir)
    shutil.copyfile(
        src='./snapshot/receive/client' + str(mydevice.device_id) + '/epoch' + str(
            mydevice.federated_epoch - 1) + '.pth',
        dst='./snapshot/final/client' + str(mydevice.device_id) + '/final.pth')
    del_file('./snapshot/send/client' + str(mydevice.device_id))
    del_file('./snapshot/receive/client' + str(mydevice.device_id))
