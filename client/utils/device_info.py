from federated_learning.network.encryption import encryption


class device_info:
    def __init__(self):
        # dataset build
        self.normalize_type = 0
        self.non_iid = 0  # 1:non-iid, 0:iid
        self.train_dataset = 'electra_modbus'  # 'CICIDS_2017'
        self.class_type = 1  # 1:muti-class, 0:binary
        self.batch_size = 65536
        self.attack_dic = {'FORCE_ERROR_ATTACK': 0, 'MITM_UNALTERED': 1, 'READ_ATTACK': 2, 'RECOGNITION_ATTACK': 3,
              'RESPONSE_ATTACK': 4, 'WRITE_ATTACK': 5, 'REPLAY_ATTACK': 6, 'NORMAL': 7}

        # model build
        self.device = 'cuda'
        self.model_type = 0  # 0:LSTM, 1:GRU
        self.optimizer_type = 1  # 0:sgd, 1:adam
        self.num_workers = 0
        self.print_frequency = 50
        self.learning_rate = 0.001
        self.weight_decay = 5e-4
        self.momentum = 0.9

        #network
        self.device_epoch = 0  # IoT_FD epoch for each communication epoch
        self.federated_epoch = 4
        self.device_id = 0
        self.key = encryption()
        self.recv_addr = '192.168.255.1'
        self.recv_port = 4000
        self.target_addr = '192.168.255.132'
        self.target_port = 4000


mydevice = device_info()
