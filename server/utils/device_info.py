class server_info:
    def __init__(self):
        # dataset build
        self.normalize_type = 0
        self.train_dataset = 'CIC_TON_IOT'  # 'CICIDS_2017'
        self.class_type = 1  # 1:muti-class, 0:binary
        self.batch_size = 65536
        self.attack_dic = {}

        # model build
        self.device = 'cpu'
        self.model_type = 1  # 0:LSTM, 1:GRU
        self.optimizer_type = 1  # 0:sgd, 1:adam
        self.num_workers = 0
        self.print_frequency = 50
        self.learning_rate = 0.001
        self.weight_decay = 5e-4
        self.momentum = 0.9

        # network
        self.device_epoch = 1  # IoT_FD epoch for each communication epoch
        self.federated_epoch = 4
        self.address = '192.168.255.1'
        self.port = 8080
        self.target = []

    def load_client(self):
        file = open('./config.txt', 'r')
        lines = file.readlines()
        for line in lines:
            client = line.split(",")
            self.target.append({"address": client[0], "port": int(client[1])})


mydevice = server_info()
mydevice.load_client()
