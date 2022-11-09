from federated_learning.utils.device_info import mydevice
from federated_learning.FL_train import train_baseline

if __name__ == '__main__':
    mydevice.recv_addr = '192.168.255.1'
    mydevice.recv_port = 4001
    mydevice.target_addr = '192.168.255.132'
    mydevice.target_port = 4000
    train_baseline()
