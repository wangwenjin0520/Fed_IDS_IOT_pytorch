import socket
import struct
import logging
import os
import hashlib
import json

logger = logging.getLogger('global')


def md5_verify(filepath, veri_code):
    m = hashlib.md5()
    file = open(filepath, 'rb')
    m.update(file.read())
    file.close()
    if m.hexdigest() == veri_code:
        return True
    else:
        return False


def socket_service_init_parameter(address, port):
    s = socket.socket()
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.bind((address, port))
    s.listen(10)
    conn, addr = s.accept()
    while True:
        buf = conn.recv(1024)
        if buf:
            result = json.loads(buf.decode('utf-8'))
            conn.sendall(bytes("200", encoding="utf-8"))
            # end, disconnect
            s.close()
            conn.close()
            break
    return result


def socket_service_init_usecolumns(address, port):
    s = socket.socket()
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.bind((address, port))
    s.listen(10)
    conn, addr = s.accept()
    while True:
        buf = conn.recv(2048)
        if buf:
            result = json.loads(buf.decode('utf-8'))
            conn.sendall(bytes("200", encoding="utf-8"))
            # end, disconnect
            s.close()
            conn.close()
            break
    return result["drop_columns"]


def socket_service_file(epoch, address, port):
    s = socket.socket()
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.bind((address, port))
    s.listen(10)
    conn, addr = s.accept()

    # start receive
    message_size = struct.calcsize('!32s')
    buf = conn.recv(message_size)
    md5_veri_code = struct.unpack('!32s', buf)
    filepath = './snapshot/epoch' + str(epoch) + '.pth'
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
            break
        else:
            conn.sendall(bytes('500', encoding='utf-8'))
            continue
    logger.info("model from server is received")
    conn.close()


