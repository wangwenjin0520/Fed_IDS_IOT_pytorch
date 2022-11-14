from torch.utils.data import Dataset
import numpy as np
import pandas as pd
import logging
logger = logging.getLogger('global')


class MyDataset(Dataset):
    def __init__(self, data, label, num_classes):
        self.num_classes = num_classes
        self.data = np.array(data)
        self.label = np.array(label)

    def __len__(self):
        return len(self.label)

    def get_featuresize(self):
        return self.data.shape[1]

    def __getitem__(self, index):
        data = self.data[index]
        label = self.label[index]
        return data, label


def normalization(data):
    eps = 2e-15
    data_sort = data.apply(lambda x: (x - np.min(x, axis=0)) /
                                     (np.max(x, axis=0) - np.min(x, axis=0) + eps)).astype(np.float32)
    return data_sort


def load_test(attack_dict):
    files_name = './dataset/dataset.csv'
    datasets = pd.read_csv(files_name)
    datasets = datasets.replace({'label': attack_dict})
    total_size = len(datasets)
    datasets_group = datasets.groupby(by='label')
    logger.info("----------------------------------------------------")
    logger.info("            testset     information                 ")
    key_list = list(attack_dict.keys())
    for key, group in datasets_group:
        logger.info("{}: {}".format(key_list[key], len(group)))
    logger.info("TOTAL_SIZE:{}".format(total_size))
    logger.info("----------------------------------------------------")
    datasets = datasets.sample(frac=1).reset_index(drop=True)
    data = normalization(datasets.drop(columns=['label']))
    label = datasets.iloc[:, len(datasets.columns) - 1]
    return data, label
