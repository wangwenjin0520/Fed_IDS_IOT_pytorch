#!coding=utf-8
import socket
import os
import logging
import struct
import json
import hashlib
logger = logging.getLogger('global')


def md5_encrypt(filepath):
    m = hashlib.md5()
    file = open(filepath, 'rb')
    m.update(file.read())
    file.close()
    return m.hexdigest()


def socket_send_init_client(message, address, port):
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


def socket_send_drop_columns(drop_columns, address, port):
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect((address, port))
    data = json.dumps({"drop_columns": drop_columns})
    while True:
        s.send(bytes(data, encoding="utf-8"))
        buf = s.recv(3)
        if buf.decode('utf-8') == "200":
            s.shutdown(2)
            s.close()
            break


def socket_send_file(address, port):
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect((address, port))
    fhead = struct.pack('!32s', md5_encrypt("./snapshot/global.pth").encode('latin-1'))
    s.send(fhead)
    # send file
    while True:
        fp_tmp = open("./snapshot/global.pth", 'rb')
        while True:
            data = fp_tmp.read(1024)
            if not data:
                s.send(bytes("EOF", encoding='utf-8'))
                break
            s.send(data)
        fp_tmp.close()
        buf = s.recv(1024)
        if buf.decode("utf-8") == "200":
            os.remove("./snapshot/global.pth")
            logger.info('model transmission completed')
            s.shutdown(2)
            s.close()
            break
        else:
            logger.info("model transmission fault")
            continue
