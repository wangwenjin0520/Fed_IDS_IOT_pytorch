import time
import torch
import logging
import os
from server.utils.server import server_info
from server.model.CNN import CNN
from server.model.GRU import GRU
from server.model.LSTM import LSTM
from server.utils.calculate import score_plot
from server.utils.log_helper import init_log, add_file_handler
from server.loss.crossentropyloss import LabelSmoothCrossEntropyLoss
from server.loss.centerloss import CenterLoss
from server.utils.memory import Monitor

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
    a = Monitor(1)
    a.start()

    # init log
    init_log('global', logging.INFO)
    add_file_handler('global', './logs/logs.txt', logging.INFO)
    logger.info("log init done")

    # init server and client
    server = server_info()
    server.load_client()
    server.init_client()

    # init dataset
    #start_time = time.time()
    #server.load_dataset()
    #logger.info("dataset build done. time:{:.4f}".format(time.time() - start_time))

'''
    # calculate importance
    for device_id in range(server.num_devices):
        start_time = time.time()
        importance = client_list[device_id].calculate_importance()
        server.importance_append(importance)
        logger.info("device {} importance finished, time:{:.4f}".format(device_id, time.time() - start_time))

    drop_columns = server.importance_calculation()
    logger.info("importance dictionary:{}".format(server.importance_dict))

    for device_id in range(server.num_devices):
        client_list[device_id].feature_reduction(drop_columns)
    server.feature_reduction(drop_columns)
    logger.info("feature reduction finished, {} features have dropped, {} features remained".format(
        len(drop_columns), server.feature_size))
    logger.info("reduction columns:{}".format(drop_columns))

    # init evaluation
    s = score_plot(server.num_devices, server.evaluation_client)

    # init model
    global_model, global_optimizer, global_criterions = criterion_init(model_type=server.model_type,
                                                                       device=server.device,
                                                                       feature_size=server.feature_size,
                                                                       num_classes=len(server.attack_dict),
                                                                       learning_rate=server.learning_rate,
                                                                       weight_decay=server.weight_decay)
    if server.fed_algorithm != 'fedavg' and server.fed_algorithm != 'fedavg+centerloss':
        tmp_model, _, _ = criterion_init(model_type=server.model_type,
                                         device=server.device,
                                         feature_size=server.feature_size,
                                         num_classes=len(server.attack_dict),
                                         learning_rate=server.learning_rate,
                                         weight_decay=server.weight_decay)
    model_state_dict = {"model": global_model.state_dict()}
    optimizer_state_dict = {"optimizer": global_optimizer.state_dict()}
    torch.save(model_state_dict, "./snapshot/after/global.pth")
    for device_id in range(server.num_devices):
        torch.save(optimizer_state_dict, "./snapshot/before/optimizer_device" + str(device_id) + ".pth")
        torch.save(model_state_dict, "./snapshot/before/model_device" + str(device_id) + ".pth")

    # IoT_FD
    logger.info("start training")
    for federated_epoch in range(server.federated_epoch):
        # iterate each device
        for device_id in range(server.num_devices):
            # init the local model and optimizer
            if server.fed_algorithm == 'fedavg' or server.fed_algorithm == 'fedavg+centerloss':
                client_list[device_id].train(model=global_model,
                                             criterions=global_criterions,
                                             optimizer=global_optimizer)
            else:
                client_list[device_id].train(model=global_model,
                                             tmp_model=tmp_model,
                                             criterions=global_criterions,
                                             optimizer=global_optimizer)

        # global aggregation
        server.aggregation()

        # global evaluation
        logger.info("-------------------------global epoch {}----------------------".format(str(federated_epoch)))
        model_state = torch.load('./snapshot/after/global.pth')
        global_model.load_state_dict(model_state['model'])
        accuracy, precision, recall, f1_score = server.evaluation(global_model)
        s.update(-1, federated_epoch, accuracy, precision, recall, f1_score)
        logger.info('--------------------------------------------------------------')

        # client evaluation
        if server.evaluation_client:
            for device_id in range(server.num_devices):
                logger.info("--------------------------device {}---------------------------".format(str(device_id)))
                model_state = torch.load('./snapshot/before/model_device' + str(device_id) + '.pth')
                global_model.load_state_dict(model_state['model'])
                accuracy, precision, recall, f1_score = server.evaluation(global_model)
                s.update(device_id, federated_epoch, accuracy, precision, recall, f1_score)
                logger.info('--------------------------------------------------------------')
    #a.stop()
    s.plot()
    s.save_result()
'''