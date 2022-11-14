import pandas as pd
import logging
import math

logger = logging.getLogger('global')


def load_data(use_columns):
    files_name = './dataset/dataset.csv'
    datasets = pd.read_csv(files_name, usecols=mydevice.use_columns)
    datasets = datasets.replace({'label': mydevice.attack_dic})
    total_size = len(datasets)
    datasets_group = datasets.groupby(by='label')
    logger.info("----------------------------------------------------")
    logger.info("               test set information                 ")
    key_list = list(mydevice.attack_dic.keys())
    for key, group in datasets_group:
        logger.info("{}: {}".format(key_list[key], len(group)))
    logger.info("TOTAL_SIZE:{}".format(total_size))
    logger.info("----------------------------------------------------")
    return datasets.sample(frac=1).reset_index(drop=True)


def process_testlabel(label_pd):
    key_list = list(mydevice.attack_dic.keys())
    label_pd["predict_label"] = label_pd["predict_label"].apply(
        lambda x: 0 if x == mydevice.attack_dic[key_list[-1]] else 1)
    label_pd["true_label"] = label_pd["true_label"].apply(lambda x: 0 if x == mydevice.attack_dic[key_list[-1]] else 1)
    return label_pd


def balanced(datasets):
    group_length = 8192
    data = datasets.groupby(by='label')
    new_datasets = pd.DataFrame()
    for key, group in data:
        if len(group) < group_length:
            group_num = int(group_length / len(group))
            remain_num = group_length - len(group) * group_num
            group_list = find_binary(group_num)
            tmp = group
            for index in range(group_list[-1] + 1):
                if index in group_list:
                    new_datasets = pd.concat([new_datasets, tmp], axis=0)
                tmp = pd.concat([tmp, tmp], axis=0)
            new_datasets = pd.concat([new_datasets, group.sample(n=remain_num).reset_index(drop=True)], axis=0)
        else:
            new_datasets = pd.concat([new_datasets, group], axis=0)
    return new_datasets.sample(frac=1).reset_index(drop=True)


def find_binary(number):
    tmp = number
    binary_list = []
    while tmp != 0:
        a = int(math.log2(tmp))
        binary_list.append(a)
        tmp = tmp - int(math.pow(2, a))
    return list(reversed(binary_list))


if __name__ == '__main__':
    from server.utils.memory import Monitor

    a = Monitor(0.01)
    a.start()

    # client
    files_name = './IoT_FD.csv'
    datasets = pd.read_csv(files_name, usecols=mydevice.use_columns)
    a.stop()
