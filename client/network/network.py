import socket
import logging
import struct
import json
import hashlib
import ssl
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


class network:
    def __init__(self, address, port, target_address, target_port):
        self.address = address
        self.port = port
        self.target_address = target_address
        self.target_port = target_port
        self.connection = None
        
    def build_connection(self):
        context = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
        context.check_hostname = False
        context.verify_mode = ssl.CERT_NONE
        tmp_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        tmp_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        tmp_socket.bind((self.address, self.port))
        tmp_socket.connect((self.target_address, self.target_port))
        self.connection = context.wrap_socket(tmp_socket)

    def socket_send_feature_importance(self, message):
        data = json.dumps(message)
        while True:
            self.connection.send(bytes(data, encoding="utf-8"))
            buf = self.connection.recv(3)
            if buf.decode('utf-8') == "200":
                break

    def socket_send_file(self, epoch):
        filepath = "./snapshot/epoch" + str(epoch) + ".pth"
        fhead = struct.pack('!32s', md5_encrypt(filepath).encode('latin-1'))
        self.connection.send(fhead)
        # send file
        while True:
            fp_tmp = open(filepath, 'rb')
            while True:
                data = fp_tmp.read(1024)
                if not data:
                    self.connection.send(bytes("EOF", encoding='utf-8'))
                    break
                self.connection.send(data)
            fp_tmp.close()
            buf = self.connection.recv(1024)
            if buf.decode("utf-8") == "200":
                logger.info('model transmission completed')
                break
            else:
                logger.info("model transmission fault")
                continue

    def socket_receive_init_parameter(self):
        while True:
            buf = self.connection.recv(1024)
            if buf:
                result = json.loads(buf.decode('utf-8'))
                self.connection.sendall(bytes("200", encoding="utf-8"))
                break
        return result

    def socket_receive_init_usecolumns(self):
        while True:
            buf = self.connection.recv(2048)
            if buf:
                result = json.loads(buf.decode('utf-8'))
                self.connection.sendall(bytes("200", encoding="utf-8"))
                break
        return result["drop_columns"]

    def socket_receive_file(self):
        # start receive
        message_size = struct.calcsize('!32s')
        buf = self.connection.recv(message_size)
        md5_veri_code = struct.unpack('!32s', buf)
        filepath = './snapshot/global.pth'
        while True:
            fp = open(filepath, 'wb')
            # receive the file
            while True:
                data = self.connection.recv(1024)
                if data[-3:] == bytes('EOF', encoding='utf-8'):
                    fp.write(data[0:-3])
                    break
                fp.write(data)
            fp.close()
            if md5_verify(filepath, md5_veri_code[0].decode('latin-1')):
                self.connection.sendall(bytes('200', encoding='utf-8'))
                break
            else:
                self.connection.sendall(bytes('500', encoding='utf-8'))
                continue
        logger.info("model from server is received")

    def close(self):
        self.connection.shutdown(2)
        self.connection.close()
