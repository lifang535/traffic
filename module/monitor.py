import os
import cv2
import time
import torch

from PIL import Image
from queue import Empty
from threading import Thread
from multiprocessing import Process, Queue
from concurrent.futures import ThreadPoolExecutor

from matplotlib import pyplot as plt

from configs import config
from request import Request

class Monitor(Process):
    def __init__(self, 
                 config: dict,
                 frame_queue: Queue,
                 car_queue: Queue,
                 person_queue: Queue,
                 end_flag):
        super().__init__()

        # ObjectDetection module
        self.frame_queue = frame_queue
        # LicenseRecognition module
        self.car_queue = car_queue
        # PersonRecognition module
        self.person_queue = person_queue
        
        self.monitor_interval = config['monitor_interval']
        
        self.qsize_path = config['qsize_path']
        
        self.end_flag = end_flag

    def run(self):
        print(f"[Monitor] Start!")
        
        frame_qsize = []
        car_qsize = []
        person_qsize = []
        
        adjust_monitor_interval = time.time()
        while not self.end_flag.value:
            frame_qsize.append(self.frame_queue.qsize())
            car_qsize.append(self.car_queue.qsize())
            person_qsize.append(self.person_queue.qsize())
            
            time.sleep(max(0, self.monitor_interval / 1000 - (time.time() - adjust_monitor_interval)))
            adjust_monitor_interval = time.time()
            
        # draw queue size
        # 创建一个包含 3 个子图的图形
        fig, axs = plt.subplots(3, 1, figsize=(10, 15))

        # 绘制第一个折线图
        axs[0].plot(frame_qsize, label='frame qsize')
        axs[0].set_title('ObjectDetection')
        axs[0].set_xlabel('Time')
        axs[0].set_ylabel('Qsize')
        axs[0].legend()

        # 绘制第二个折线图
        axs[1].plot(car_qsize, label='car qsize', color='orange')
        axs[1].set_title('LicenseRecognition')
        axs[1].set_xlabel('Time')
        axs[1].set_ylabel('Qsize')
        axs[1].legend()

        # 绘制第三个折线图
        axs[2].plot(person_qsize, label='person qsize', color='green')
        axs[2].set_title('PersonRecognition')
        axs[2].set_xlabel('Time')
        axs[2].set_ylabel('Qsize')
        axs[2].legend()

        # 调整子图之间的间距
        plt.tight_layout()

        # 保存图形为 PDF 文件
        plt.savefig(self.qsize_path)
        print(f"[Monitor] saved qsize plot to {self.qsize_path}")
        
        # 关闭图形
        plt.close()
        
    def _end(self):
        self.end_flag = True
        print(f"[Monitor] Stop!")
