import hashlib
import json
import socket
import struct
import time
import logging
import os
from threading import Thread

logger = logging.getLogger('global')


def md5_encrypt(filepath):
    m = hashlib.md5()
    file = open(filepath, 'rb')
    m.update(file.read())
    file.close()
    return m.hexdigest()


def md5_verify(filepath, veri_code):
    m = hashlib.md5()
    file = open(filepath, 'rb')
    m.update(file.read())
    file.close()
    if m.hexdigest() == veri_code:
        return True
    else:
        return False


message_list = []


def deal_file(conn, key):
    message_size = struct.calcsize('!32s')
    buf = conn.recv(message_size)
    md5_veri_code = struct.unpack('!32s', buf)
    filepath = './snapshot/' + key.replace(":",",") + '.pth'
    while True:
        fp = open(filepath, 'wb')
        # receive the file
        while True:
            data = conn.recv(1024)
            if data[-3:] == bytes('EOF', encoding='utf-8'):
                fp.write(data[0:-3])
                break
            fp.write(data)
        fp.close()
        if md5_verify(filepath, md5_veri_code[0].decode('latin-1')):
            conn.sendall(bytes('200', encoding='utf-8'))
            message_list.append(1)
            break
        else:
            conn.sendall(bytes('500', encoding='utf-8'))
            continue
    logger.info("model from client " + key + " is received")


def deal_feature_selection(conn):
    while True:
        buf = conn.recv(2048)
        if buf:
            message_list.append(json.loads(buf.decode('utf-8')))
            break
    conn.sendall(bytes('200', encoding='utf-8'))


class network:
    def __init__(self, address, port):
        self.target = {}
        self.address = address
        self.port = port

    def load_client(self):
        file = open('./utils/config.txt', 'r')
        lines = file.readlines()
        for (index, line) in enumerate(lines):
            client = line.split(",")
            name = client[0] + ":" + client[1]
            self.target.update({name: {"id": index}})
        s = socket.socket()
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind((self.address, self.port))
        s.listen(10)
        for i in range(len(self.target)):
            conn, addr = s.accept()
            client_name = str(addr[0]) + ":" + str(addr[1])
            if client_name in self.target.keys():
                self.target[client_name].update({"connection": conn})
        s.close()

    def socket_receive_feature_selection(self):
        message_list.clear()
        for key, value in self.target.items():
            self.target[key]["state"] = 0
            thread = Thread(
                target=deal_feature_selection,
                args=(self.target[key]["connection"],)
            )
            thread.start()
        while True:
            if len(message_list) == len(self.target):
                break
            else:
                time.sleep(5)
        return message_list

    def socket_receive_file(self):
        message_list.clear()
        for key, value in self.target.items():
            self.target[key]["state"] = 0
            thread = Thread(
                target=deal_file,
                args=(self.target[key]["connection"], key)
            )
            thread.start()
        while True:
            if len(message_list) == len(self.target):
                break
            else:
                time.sleep(5)

    def socket_send_init_client(self, message):
        data = json.dumps(message)
        for key, value in self.target.items():
            while True:
                value["connection"].send(bytes(data, encoding="utf-8"))
                buf = value["connection"].recv(3)
                if buf.decode('utf-8') == "200":
                    break

    def socket_send_drop_columns(self, drop_columns):
        data = json.dumps({"drop_columns": drop_columns})
        for key, value in self.target.items():
            while True:
                value["connection"].send(bytes(data, encoding="utf-8"))
                buf = value["connection"].recv(3)
                if buf.decode('utf-8') == "200":
                    break

    def socket_send_file(self):
        for key, value in self.target.items():
            fhead = struct.pack('!32s', md5_encrypt("./snapshot/global.pth").encode('latin-1'))
            value["connection"].send(fhead)
            # send file
            while True:
                fp_tmp = open("./snapshot/global.pth", 'rb')
                while True:
                    data = fp_tmp.read(1024)
                    if not data:
                        value["connection"].send(bytes("EOF", encoding='utf-8'))
                        break
                    value["connection"].send(data)
                fp_tmp.close()
                buf = value["connection"].recv(1024)
                if buf.decode("utf-8") == "200":
                    os.remove("./snapshot/global.pth")
                    logger.info('model transmission completed')
                    break
                else:
                    logger.info("model transmission fault")
                    continue

    def close(self):
        for key, value in self.target.items():
            value["connection"].shutdown(2)
            value["connection"].close()
