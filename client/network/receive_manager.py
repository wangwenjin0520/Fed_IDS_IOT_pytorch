import socket
import struct
import logging
import os
import json

logger = logging.getLogger('global')


def socket_service_file(epoch, address, port):
    s = socket.socket()
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.bind((address, port))
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
