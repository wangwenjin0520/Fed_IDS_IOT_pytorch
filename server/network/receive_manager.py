import hashlib
import socket
import struct
import logging
import os
import time
import json
from threading import Thread

logger = logging.getLogger('global')
receive_list = []


def md5_verify(filepath, veri_code):
    m = hashlib.md5()
    file = open(filepath, 'rb')
    m.update(file.read())
    file.close()
    if m.hexdigest() == veri_code:
        return True
    else:
        return False


def socket_receive_feature_selection(address, port, target):
    receive_list.clear()
    s = socket.socket()
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.bind((address, port))
    s.listen(10)
    while len(receive_list) != len(target):
        conn, addr = s.accept()
        receive_list.append({"address": str(addr[0]), "port": str(addr[1]), "state": 0, "message": None})
        thread = Thread(target=deal_feature_selection, args=(conn, str(addr[0]), str(addr[1])))
        thread.start()
    for i in receive_list:
        while True:
            if i["state"] == 1:
                break
            else:
                time.sleep(5)
    result = []
    for i in receive_list:
        result.append(i["message"])
    return result


def deal_feature_selection(conn, addr, port):
    while True:
        buf = conn.recv(2048)
        if buf:
            for i in receive_list:
                if i["address"] == addr and i["port"] == port:
                    i["state"] = 1
                    i["message"] = json.loads(buf.decode('utf-8'))
                    break
            break
    conn.sendall(bytes('200', encoding='utf-8'))
    conn.close()


def socket_receive_file(epoch):
    s = socket.socket()
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.bind((mydevice.recv_addr, mydevice.recv_port))
    s.listen(10)
    conn, addr = s.accept()

    # start receive
    message_size = struct.calcsize('!il32s')
    buf = conn.recv(message_size)
    _, filesize, md5_veri_code = struct.unpack('!il32s', buf)
    path_dir = './snapshot/receive/client' + str(mydevice.device_id)
    if not os.path.exists(path_dir):
        os.mkdir(path_dir)
    filepath = path_dir + '/epoch' + str(epoch) + '.pth'
    while True:
        fp = open(filepath + '.txt', 'wb')
        # receive the file
        while True:
            data = conn.recv(1024)
            if data[-3:] == bytes('EOF', encoding='utf-8'):
                fp.write(data[0:-3])
                break
            fp.write(data)
        fp.close()
        if mydevice.key.md5_verify(filepath + '.txt', md5_veri_code.decode('latin-1')):
            conn.sendall(bytes('model received', encoding='utf-8'))
            break
        else:
            conn.sendall(bytes('model broken', encoding='utf-8'))
            continue
    mydevice.key.decrypt(filepath)
    logger.info("model from server is received")
    conn.close()


def socket_receive_init():
    s = socket.socket()
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.bind((mydevice.address, mydevice.port))
    s.listen(10)
    conn, addr = s.accept()
    while True:
        # ------------------------------------------------------
        # 3i:normalize type(0, 1, 2)
        # classes type(0:binary, 1:multi), global epoch
        message_size = struct.calcsize('!7i')
        buf = conn.recv(message_size)
        if buf:
            mydevice.device_id, \
            mydevice.model_type, \
            mydevice.normalize_type, \
            mydevice.batch_size, \
            mydevice.non_iid, \
            mydevice.federated_epoch, \
            mydevice.device_epoch = struct.unpack('!7i', buf)
            break
    while True:
        buf = conn.recv(1024)
        if buf:
            conn.sendall(mydevice.key.encrypt_aes_key(buf))
            # end, disconnect
            s.close()
            conn.close()
            break
