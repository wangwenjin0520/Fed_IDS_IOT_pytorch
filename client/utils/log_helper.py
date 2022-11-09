import os
import logging
logs = set()


class Filter:
    def __init__(self, flag):
        self.flag = flag

    def filter(self, x):
        return self.flag


def get_format(logger, level):
    if 'RANK' in os.environ:
        rank = int(os.environ['RANK'])

        if level == logging.INFO:
            logger.addFilter(Filter(rank == 0))
    else:
        rank = 0
    format_str = '[%(asctime)s-rk{}-%(filename)s#%(lineno)3d] %(message)s'.format(rank)
    formatter = logging.Formatter(format_str)
    return formatter


def get_format_custom(logger, level):
    if 'RANK' in os.environ:
        rank = int(os.environ['RANK'])
        if level == logging.INFO:
            logger.addFilter(Filter(rank == 0))
    else:
        rank = 0
    format_str = '[%(asctime)s-rk{}-%(message)s'.format(rank)
    formatter = logging.Formatter(format_str)
    return formatter


def init_log(name, level=logging.INFO, format_func=get_format):
    if (name, level) in logs:
        return
    logs.add((name, level))
    logger = logging.getLogger(name)
    logger.setLevel(level)
    ch = logging.StreamHandler()
    ch.setLevel(level)
    formatter = format_func(logger, level)
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    return logger


def add_file_handler(name, log_file, level=logging.INFO):
    logger = logging.getLogger(name)
    fh = logging.FileHandler(log_file)
    fh.setFormatter(get_format(logger, level))
    logger.addHandler(fh)

