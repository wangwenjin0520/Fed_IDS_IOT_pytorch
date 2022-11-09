from Crypto.Cipher import AES
from Crypto.PublicKey import RSA
from Crypto.Cipher import PKCS1_v1_5
import random
import hashlib
import os


class encryption:
    def __init__(self):
        self.aes_key = bytes([random.randint(0, 255) for i in range(32)])
        self.length = 16
        self.unpad = lambda date: date[0:-date[-1]]

    def encrypt_aes_key(self, key):
        public_key = RSA.importKey(key)
        cipher = PKCS1_v1_5.new(public_key)
        data = cipher.encrypt(self.aes_key)
        return data

    def pad(self, text):
        add = self.length - (len(text) % self.length)
        cipher_text = text + (bytes([add]) * add)
        return cipher_text

    def encrypt(self, filepath):
        cipher = AES.new(self.aes_key, AES.MODE_ECB)
        fp = open(filepath, 'rb')
        fp_tmp = open(filepath + '.txt', 'wb')
        while True:
            data = fp.read(1020)
            if not data:
                break
            cipher_text = self.pad(data)
            cipher_text = cipher.encrypt(cipher_text)
            fp_tmp.write(cipher_text)
        fp.close()
        fp_tmp.close()
        md5_encrypt = self.md5_encrypt(filepath + '.txt')
        file_size = os.stat(filepath + '.txt').st_size
        return file_size, md5_encrypt

    def decrypt(self, filepath):
        cipher = AES.new(self.aes_key, AES.MODE_ECB)
        fp_tmp = open(filepath+'.txt', 'rb')
        fp = open(filepath, 'wb')
        plain_text = ""
        while True:
            data = fp_tmp.read(1024)
            if not data:
                break
            plain_text = cipher.decrypt(data)
            fp.write(self.unpad(plain_text))
        fp.close()
        fp_tmp.close()
        os.remove(filepath+'.txt')
        return self.unpad(plain_text)

    def md5_encrypt(self, filepath):
        m = hashlib.md5()
        file = open(filepath, 'rb')
        m.update(file.read())
        file.close()
        return m.hexdigest()

    def md5_verify(self, filepath, veri_code):
        m = hashlib.md5()
        file = open(filepath, 'rb')
        m.update(file.read())
        file.close()
        if m.hexdigest() == veri_code:
            return True
        else:
            return False
