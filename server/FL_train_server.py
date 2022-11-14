import time
import torch
import logging
import os

from server.utils.calculate import score_plot
from server.utils.server import server_info
from server.utils.log_helper import init_log, add_file_handler
from server.utils.memory import Monitor
logger = logging.getLogger('global')

if __name__ == '__main__':
    #a = Monitor(1)
    #a.start()

    # init log
    init_log('global', logging.INFO)
    add_file_handler('global', './logs/logs.txt', logging.INFO)
    logger.info("log init done")

    # init server and client
    server = server_info()
    server.load_client()
    server.init_client()

    # calculate importance
    server.load_dataset()
    server.feature_reduction()

    # init client model
    server.init_client_model()

    # IoT_FD
    logger.info("start training")
    server.aggregation()
