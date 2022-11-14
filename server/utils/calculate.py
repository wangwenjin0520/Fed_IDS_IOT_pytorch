import numpy as np
import pandas as pd
import logging
import matplotlib.pyplot as pyplot
import time
import torch
#from server.utils.server import server_info
from server.dataset.data_processing import process_testlabel

logger = logging.getLogger('global')


class label_record:
    def __init__(self, label_name):
        self.label_name = label_name
        self.true_positive = 0
        self.true_negative = 0
        self.false_positive = 0
        self.false_negative = 0
        self.volumn = 0

    def update(self, TP, TN, FP, FN):
        self.true_positive = TP
        self.true_negative = TN
        self.false_positive = FP
        self.false_negative = FN

    def count(self, volumn):
        self.volumn += volumn


class score:
    def __init__(self, attack_dic):
        self.attack_dic = {}
        for key, value in attack_dic.items():
            self.attack_dic.update({value: label_record(key)})
        self.num_classes = len(self.attack_dic)
        self.class_matrix = np.zeros([self.num_classes, self.num_classes]).astype(np.int)
        self.dataset_volumn = 0

    def update_class(self):
        self.dataset_volumn = np.sum(np.concatenate(self.class_matrix))
        for class_index in range(self.num_classes):
            # true positive
            TP = self.class_matrix[class_index][class_index].item()

            # false nagative
            FN = 0
            for i in range(self.num_classes):
                FN += self.class_matrix[i][class_index].item()
            FN -= TP

            # false positive
            FP = 0
            for i in range(self.num_classes):
                FP += self.class_matrix[class_index][i].item()
            FP -= TP

            # true nagative
            TN = self.dataset_volumn - TP - FN - FP
            self.attack_dic[class_index].update(TP, TN, FP, FN)

    def accuracy(self):
        true = 0
        for key, value in self.attack_dic.items():
            true += value.true_positive
        return true / self.dataset_volumn * 100

    def precision(self):
        precision_dic = {}
        total_precision = 0
        for key, value in self.attack_dic.items():
            true_positive = value.true_positive
            false_positive = value.false_positive
            if true_positive + false_positive == 0:
                precision_value = 0
            else:
                precision_value = round(true_positive / (true_positive + false_positive), 4)
            precision_dic.update({value.label_name: precision_value})
            total_precision += precision_value * value.volumn / self.dataset_volumn
        return precision_dic, total_precision

    def recall(self):
        recall_dic = {}
        total_recall = 0
        for key, value in self.attack_dic.items():
            true_positive = self.attack_dic[key].true_positive
            false_negative = self.attack_dic[key].false_negative
            if true_positive + false_negative == 0:
                recall_value = 0
            else:
                recall_value = round(true_positive / (true_positive + false_negative), 4)
            recall_dic.update({value.label_name: recall_value})
            total_recall += recall_value * self.attack_dic[key].volumn / self.dataset_volumn
        return recall_dic, total_recall

    def compute(self):
        self.update_class()
        accuracy = self.accuracy()
        precision_dic, total_precision = self.precision()
        recall_dic, total_recall = self.recall()
        f1_score = 2 * total_precision * total_recall / (total_precision + total_recall)
        logger.info("precision:{}".format(precision_dic))
        logger.info("recall:   {}".format(recall_dic))
        return accuracy, total_precision, total_recall, f1_score

    def update(self, predict_label, true_label):
        label_pd = pd.DataFrame({"predict_label": pd.Series(predict_label.tolist()),
                                 "true_label": pd.Series(true_label.tolist())})
        label_pd_group = label_pd.groupby(['predict_label', 'true_label'])
        for (predict_label_key, true_label_key), value in label_pd_group:
            predict_index = self.find_index(predict_label_key)
            true_index = self.find_index(true_label_key)
            self.class_matrix[predict_index][true_index] += len(value)
            self.attack_dic[true_label_key].count(len(value))

    def find_index(self, label):
        key_list = list(self.attack_dic.keys())
        return key_list.index(label)


class score_plot:
    def __init__(self, target):
        self.score_dic = {}
        self.score_dic.update({"global": {"accuracy": [], "precision": [], "recall": [], "f1_score": []}})
        for key in target.keys():
            self.score_dic.update({key: {"accuracy": [], "precision": [], "recall": [], "f1_score": []}})
        self.epoch = []

    def update(self, name, epoch, accuracy, precision, recall, f1_score):
        if epoch not in self.epoch:
            self.epoch.append(epoch)
        self.score_dic[name]["accuracy"].append(accuracy)
        self.score_dic[name]["precision"].append(precision)
        self.score_dic[name]["recall"].append(recall)
        self.score_dic[name]["f1_score"].append(f1_score)

    def plot(self):
        pyplot.figure()
        color_list = get_cmap(len(self.score_dic))
        # accuracy
        plot1 = pyplot.subplot(111)
        pyplot.xlabel("epoch", fontsize=18)
        pyplot.ylabel("accuracy", fontsize=18)
        for index, (key, value) in enumerate(self.score_dic.items()):
            pyplot.plot(self.epoch, value["accuracy"], label=key, color=color_list(index))
        pyplot.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0, fontsize=18)
        pyplot.tick_params(labelsize=13)
        pyplot.show()

        accuracy_result = {}
        for key, value in self.score_dic.items():
            if key == -1:
                for key_sub, value_sub in value.items():
                    accuracy_result.update({"global_model"+key_sub: value_sub})
            else:
                for key_sub, value_sub in value.items():
                    accuracy_result.update({"device"+str(key)+key_sub: value_sub})

        result = pd.DataFrame(accuracy_result)
        result.to_excel('./logs/result.xlsx', index=False)


def get_cmap(n, name='hsv'):
    """Returns a function that maps each index in 0, 1, ..., n-1 to a distinct
    RGB color; the keyword argument name must be a standard mpl colormap name."""
    return pyplot.cm.get_cmap(name, n)
