#!coding=utf-8
import socket
import os
import logging
import struct
import json
logger = logging.getLogger('global')


def socket_send_feature_importance(message, address, port):
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect((address, port))
    data = json.dumps(message)
    while True:
        s.send(bytes(data, encoding="utf-8"))
        buf = s.recv(3)
        if buf.decode('utf-8') == "200":
            s.shutdown(2)
            s.close()
            break

def socket_client(filepath):
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect((mydevice.target_addr, mydevice.target_port))
    file_size, md5_encrypt = mydevice.key.encrypt(filepath)
    fhead = struct.pack('!il32s', mydevice.device_id, file_size, md5_encrypt.encode('latin-1'))
    s.send(fhead)

    # send file
    while True:
        fp_tmp = open(filepath + '.txt', 'rb')
        while True:
            data = fp_tmp.read(1024)
            if not data:
                s.send(bytes("EOF", encoding='utf-8'))
                break
            s.send(data)
        fp_tmp.close()
        buf = s.recv(1024)
        if buf.decode() == "model received":
            os.remove(filepath + '.txt')
            logger.info('model transmission completed')
            s.shutdown(2)
            s.close()
            break
        else:
            logger.info("model transmission fault")
            continue
