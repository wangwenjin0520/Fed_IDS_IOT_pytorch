import GPUtil
import time
from threading import Thread
import psutil
import os


class Monitor(Thread):
    def __init__(self, delay):
        super(Monitor, self).__init__()
        self.stopped = False
        self.delay = delay
        self.pid = os.getpid()
        self.gpu_utilization = 0
        self.memory_utilization = 0

    def run(self):
        p = psutil.Process(self.pid)
        while not self.stopped:
            gpus = GPUtil.getGPUs()
            for gpu in gpus:
                gpu_utilization = gpu.memoryUtil * 100
                if gpu_utilization >= self.gpu_utilization:
                    self.gpu_utilization = gpu_utilization
            memory_utilization = p.memory_percent()
            if memory_utilization >= self.memory_utilization:
                self.memory_utilization = memory_utilization
            #time.sleep(self.delay)

    def stop(self):
        f = open("./logs/memory.txt", "a+")
        f.write("--------------------------------------------\n")
        f.write("gpu_utilization_max:"+str(self.gpu_utilization)+"%\n")
        f.write("memory_utilization_max:"+str(self.memory_utilization)+"%\n")
        f.close()
        self.stopped = True
