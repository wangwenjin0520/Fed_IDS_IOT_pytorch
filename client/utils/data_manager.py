from torch.utils.data import Dataset, DataLoader
import numpy as np
from federated_learning.datasets.electra_modbus.data_processing import load_data
from federated_learning.utils.device_info import mydevice


class MyDataset(Dataset):
    def __init__(self, datasets, num_classes):
        self.num_classes = num_classes
        data = normalization(datasets.drop(columns=['label']))
        label = datasets.iloc[:, len(datasets.columns) - 1]
        self.data = np.array(data)
        self.label = np.array(label)
        self.feature_size = self.data.shape[1]

    def __len__(self):
        return len(self.label)

    def __getitem__(self, index):
        data = self.data[index]
        label = self.label[index]
        return data, label


def normalization(data):
    eps = 2e-15
    if mydevice.normalize_type == 0:
        data_sort = data.apply(lambda x: (x - np.min(x, axis=0)) /
                                         (np.max(x, axis=0) - np.min(x, axis=0) + eps)).astype(np.float32)
    elif mydevice.normalize_type == 1:
        data_sort = data.apply(lambda x: 1 / (1 + np.exp(-float(x)))).astype(np.float32)

    elif mydevice.normalize_type == 2:
        data_sort = data.apply(lambda x: (x - np.mean(x, axis=0)) /
                                         (np.max(x, axis=0) - np.min(x, axis=0) + eps)).astype(np.float32)
    else:
        data_sort = data.apply(lambda x: (x - np.mean(x, axis=0)) /
                                         np.std(x, axis=0)).astype(np.float32)

    return data_sort


def build_data_loader():
    datasets = load_data()
    train_set = MyDataset(datasets, len(mydevice.attack_dic))

    train_loader = DataLoader(dataset=train_set,
                              batch_size=mydevice.batch_size,
                              shuffle=True,
                              num_workers=mydevice.num_workers,
                              drop_last=True)
    return train_loader, train_set.feature_size
