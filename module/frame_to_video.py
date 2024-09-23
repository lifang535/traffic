import os
import cv2
import time
import torch

from queue import Empty
from threading import Thread
from multiprocessing import Process, Queue

from matplotlib import pyplot as plt

from configs import config
from request import Request

class FrameToVideo(Process):
    def __init__(self, 
                 config: dict,
                 car_frame_queue: Queue,
                 person_frame_queue: Queue,
                 one_by_one: Queue = None):
        super().__init__()
        
        self.output_video_dir = config['output_video_dir']
        
        # from license_recognition module
        self.car_frame_queue = car_frame_queue
        # from person_recognition module
        self.person_frame_queue = person_frame_queue
        
        # only used to send end signal
        self.frame_queue = Queue()
        self.video_queue = Queue()
        
        # save the frames of each video
        self.video_dict = {}
        self.video_saved_set = set()
        
        # save the process time of each frame
        self.process_time_dict = {} # add
        self.times = {} # add
        self.flops = {} # add
        self.car_number = {}
        self.person_number = {}
        
        self.latency_path = config['latency_path']
        self.times_path = config['times_path']
        self.flops_path = config['flops_path']

        self.end_flag = False
        
        self.one_by_one = one_by_one

    def run(self):
        print(f"[FrameToVideo] Start!")
        
        car_get_thread = Thread(target=self.car_get)
        person_get_thread = Thread(target=self.person_get)
        frame_get_thread = Thread(target=self.frame_get)
        video_get_thread = Thread(target=self.video_get)
        
        car_get_thread.start()
        person_get_thread.start()
        frame_get_thread.start()
        video_get_thread.start()

        try:
            car_get_thread.join()
            person_get_thread.join()
            frame_get_thread.join()
            video_get_thread.join()
        except KeyboardInterrupt:
            pass
        
        print(f"[FrameToVideo] Stop!")
            
    def car_get(self):
        while not self.end_flag:
            try:
                request = self.car_frame_queue.get(timeout=1)
            except Empty:
                continue
            
            if request is None:
                self.frame_queue.put(None)
                break
            
            # print(f"[FrameToVideo] video_id: {request.video_id}, frame_id: {request.frame_id}, car_id: {request.car_id}")
            
            video_id = request.video_id
            frame_id = request.frame_id
            car_id = request.car_id
            
            if video_id not in self.video_dict:
                self.video_dict[video_id] = {}
            if frame_id not in self.video_dict[video_id]:
                self.video_dict[video_id][frame_id] = {}
                self.video_dict[video_id][frame_id]['request'] = request
                
                self.video_dict[video_id][frame_id]['car'] = {}
                self.video_dict[video_id][frame_id]['person'] = {}
                self.video_dict[video_id][frame_id]['times'] = {'od': request.times[0] - request.start_time, 'lr': request.times[1] - request.start_time, 'pr': 0}
                self.video_dict[video_id][frame_id]['flops'] = {'od': request.flops[0], 'lr': 0, 'pr': 0}
            
            self.video_dict[video_id][frame_id]['times']['lr'] = request.times[1] - request.start_time
            self.video_dict[video_id][frame_id]['flops']['lr'] += request.flops[1]
            
            if request.box is not None:
                self.video_dict[video_id][frame_id]['car'][car_id] = {'box': request.box, 'label': request.label}
            
            # if len(self.video_dict[video_id]) == request.frame_number: # 删掉这行是为了记录 self.process_time_dict[frame_id]（认为顺序发送不可能乱序到达下游模块）
            self.check_video(video_id)
            
    def person_get(self):
        while not self.end_flag:
            try:
                request = self.person_frame_queue.get(timeout=1)
            except Empty:
                continue
            
            if request is None:
                self.frame_queue.put(None)
                break
            
            # print(f"[FrameToVideo] video_id: {request.video_id}, frame_id: {request.frame_id}, person_id: {request.person_id}")
            
            video_id = request.video_id
            frame_id = request.frame_id
            person_id = request.person_id
            
            if video_id not in self.video_dict:
                self.video_dict[video_id] = {}
            if frame_id not in self.video_dict[video_id]:
                self.video_dict[video_id][frame_id] = {}
                self.video_dict[video_id][frame_id]['request'] = request
                
                self.video_dict[video_id][frame_id]['car'] = {}
                self.video_dict[video_id][frame_id]['person'] = {}
                self.video_dict[video_id][frame_id]['times'] = {'od': request.times[0] - request.start_time, 'lr': 0, 'pr': request.times[1] - request.start_time}
                self.video_dict[video_id][frame_id]['flops'] = {'od': request.flops[0], 'lr': 0, 'pr': 0}
                
            self.video_dict[video_id][frame_id]['times']['pr'] = request.times[1] - request.start_time
            self.video_dict[video_id][frame_id]['flops']['pr'] += request.flops[1]
            
            if request.box is not None:
                self.video_dict[video_id][frame_id]['person'][person_id] = {'box': request.box, 'label': request.label}

            # if len(self.video_dict[video_id]) == request.frame_number: # 删掉这行是为了记录 self.process_time_dict[frame_id]（认为顺序发送不可能乱序到达下游模块）
            self.check_video(video_id)
    
    # Check if the video contains all the frames, and if the frames contain all the cars and persons
    # If so, save the video as a video file
    def check_video(self, video_id):
        # print(f"[FrameToVideo] Check video: {video_id}")
        frame_number = self.video_dict[video_id][0]['request'].frame_number
        
        for frame_id in range(frame_number):
            if frame_id not in self.video_dict[video_id]:
                return
            request = self.video_dict[video_id][frame_id]['request']
            car_number = request.car_number
            person_number = request.person_number
            
            if len(self.video_dict[video_id][frame_id]['car']) != car_number:
                return
            if len(self.video_dict[video_id][frame_id]['person']) != person_number:
                return
            
            if frame_id not in self.process_time_dict:
                self.process_time_dict[frame_id] = time.time() - request.start_time # add，TODO: 考虑顺序
                self.car_number[frame_id] = car_number
                self.person_number[frame_id] = person_number
        
        if video_id not in self.video_saved_set:
            self.video_saved_set.add(video_id)
            self.save_video(video_id)
        
    def save_video(self, video_id):
        print(f"[FrameToVideo] Save video: {video_id}")
        request = self.video_dict[video_id][0]['request']
        frame_size = request.data.shape[:2]
        video_fps = request.video_fps
        print(f"[FrameToVideo] Frame size: {frame_size}")
        
        # draw video with boxes and labels
        output_video_path = os.path.join(self.output_video_dir, f"{video_id}.mp4")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v') # (*'XVID') (*'mp4v')
        video = cv2.VideoWriter(output_video_path, fourcc, video_fps, (frame_size[1], frame_size[0]))
        
        for frame_id in range(len(self.video_dict[video_id])):
            # print(f"[FrameToVideo] Frame: {frame_id}")
            request = self.video_dict[video_id][frame_id]['request']
            frame_array = request.data # (height, width, channel), BGR
            # frame = cv2.cvtColor(frame_array, cv2.COLOR_BGR2RGB)
            frame = frame_array
            
            for car_id, car in self.video_dict[video_id][frame_id]['car'].items():
                box = car['box']
                label = car['label']
                
                # Relative coordinates need to be converted to absolute coordinates
                x1, y1, x2, y2 = box
                # x1 = int(x1 * frame_size[1]) # lifang535 remove
                # y1 = int(y1 * frame_size[0])
                # x2 = int(x2 * frame_size[1])
                # y2 = int(y2 * frame_size[0])
                x1 = int(x1) # lifang535 add
                y1 = int(y1)
                x2 = int(x2)
                y2 = int(y2)
                
                # print(f"[FrameToVideo] Car box: ({x1}, {y1}, {x2}, {y2}) label {label}")
                
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(frame, label, (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            for person_id, person in self.video_dict[video_id][frame_id]['person'].items():
                box = person['box']
                label = person['label']
                
                # Relative coordinates need to be converted to absolute coordinates
                x1, y1, x2, y2 = box
                # x1 = int(x1 * frame_size[1]) # lifang535 remove
                # y1 = int(y1 * frame_size[0])
                # x2 = int(x2 * frame_size[1])
                # y2 = int(y2 * frame_size[0])
                x1 = int(x1) # lifang535 add
                y1 = int(y1)
                x2 = int(x2)
                y2 = int(y2)
                
                # print(f"[FrameToVideo] Person box: ({x1}, {y1}, {x2}, {y2}) label {label}")
                
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, label, (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            video.write(frame)
            
        video.release()
        
        print(f"[FrameToVideo] Save video: {video_id} done! video_dict.keys(): {self.video_dict.keys()}")
        self.times = { # add
            'od': [self.video_dict[video_id][frame_id]['times']['od'] for frame_id in range(len(self.video_dict[video_id]))],
            'lr': [self.video_dict[video_id][frame_id]['times']['lr'] for frame_id in range(len(self.video_dict[video_id]))],
            'pr': [self.video_dict[video_id][frame_id]['times']['pr'] for frame_id in range(len(self.video_dict[video_id]))],
        }
        self.flops = { # add
            'od': [self.video_dict[video_id][frame_id]['flops']['od'] for frame_id in range(len(self.video_dict[video_id]))],
            'lr': [self.video_dict[video_id][frame_id]['flops']['lr'] for frame_id in range(len(self.video_dict[video_id]))],
            'pr': [self.video_dict[video_id][frame_id]['flops']['pr'] for frame_id in range(len(self.video_dict[video_id]))],
        }
        self.video_dict.pop(video_id)
        
        pass
            
    def frame_get(self):
        end_count = 0
        
        while not self.end_flag:
            try:
                frame = self.frame_queue.get(timeout=1)
            except Empty:
                continue
            
            if frame is None:
                end_count += 1
                if end_count == 2:
                    self.video_queue.put(None)
                    
                    # self._end() remove
                    break
                continue # add
    
    def video_get(self):
        while not self.end_flag:
            try:
                video = self.video_queue.get(timeout=1)
            except Empty:
                continue
            
            if video is None:
                self._end()
                break
            
    def draw_latency(self): # add，会调用两次
        process_time_list = [self.process_time_dict[frame_id] for frame_id in range(len(self.process_time_dict))][2:]
        car_number_list = [self.car_number[frame_id] for frame_id in range(len(self.car_number))][2:]
        person_number_list = [self.person_number[frame_id] for frame_id in range(len(self.person_number))][2:]
        
        print(f"[FrameToVideo] process_time_list = {process_time_list}")
        print(f"[FrameToVideo] car_number_list = {car_number_list}")
        print(f"[FrameToVideo] person_number_list = {person_number_list}")
        
        # 创建一个包含 3 个子图的图形
        fig, axs = plt.subplots(3, 1, figsize=(10, 15))

        # 绘制第一个折线图
        axs[0].plot(process_time_list, label='Process Time')
        axs[0].set_title('Process Time Over Frames')
        axs[0].set_xlabel('Frame')
        axs[0].set_ylabel('Process Time')
        axs[0].legend()

        # 绘制第二个折线图
        axs[1].plot(car_number_list, label='Car Number', color='orange')
        axs[1].set_title('Car Number Over Frames')
        axs[1].set_xlabel('Frame')
        axs[1].set_ylabel('Car Number')
        axs[1].legend()

        # 绘制第三个折线图
        axs[2].plot(person_number_list, label='Person Number', color='green')
        axs[2].set_title('Person Number Over Frames')
        axs[2].set_xlabel('Frame')
        axs[2].set_ylabel('Person Number')
        axs[2].legend()

        # 调整子图之间的间距
        plt.tight_layout()

        # 保存图形为 PDF 文件
        # latency_path = '../picture/latency/before_attacking.png'
        plt.savefig(self.latency_path)
        print(f"[FrameToVideo] saved latency plot to {self.latency_path}")
        
        # 关闭图形
        plt.close()
    
        # plt.plot(range(len(process_time_list)), process_time_list)
        # plt.savefig('../latency/2.pdf')
    
    def draw_times(self):
        print(f"[FrameToVideo] self.times = {self.times}")
        # 画在一张图上
        # 创建一个图形
        plt.figure(figsize=(10, 5))
        
        # 绘制折线图
        plt.plot(self.times['od'][2:], label='Object Detection')
        plt.plot(self.times['lr'][2:], label='License Recognition')
        plt.plot(self.times['pr'][2:], label='Person Recognition')
        
        # 添加标题和标签
        plt.title('Inference Time Over Frames')
        plt.xlabel('Frame')
        plt.ylabel('Inference Time')
        
        # 添加图例
        plt.legend()

        # 保存图形为 PDF 文件
        plt.savefig(self.times_path)
        print(f"[FrameToVideo] saved times plot to {self.times_path}")
        
        # 关闭图形
        plt.close()
        
    # def draw_flops(self):
    #     print(f"[FrameToVideo] self.flops = {self.flops}")
    #     # 画在一张图上
    #     # 创建一个图形
    #     plt.figure(figsize=(10, 5))
        
    #     # 绘制折线图
    #     plt.plot(self.flops['od'], label='Object Detection')
    #     plt.plot(self.flops['lr'], label='License Recognition')
    #     plt.plot(self.flops['pr'], label='Person Recognition')
        
    #     # 添加标题和标签
    #     plt.title('FLOPs Over Frames')
    #     plt.xlabel('Frame')
    #     plt.ylabel('FLOPs')
        
    #     # 添加图例
    #     plt.legend()

    #     # 保存图形为 PDF 文件
    #     plt.savefig(self.flops_path)
    #     print(f"[FrameToVideo] saved flops plot to {self.flops_path}")
        
    #     # 关闭图形
    #     plt.close()
        
    def draw_flops(self):
        fig, axs = plt.subplots(3, 1, figsize=(10, 15))
        
        axs[0].plot(self.flops['od'], label='Object Detection')
        axs[0].set_title('FLOPs Over Frames')
        axs[0].set_xlabel('Frame')
        axs[0].set_ylabel('FLOPs')
        axs[0].legend()
        
        axs[1].plot(self.flops['lr'], label='License Recognition', color='orange')
        axs[1].set_title('FLOPs Over Frames')
        axs[1].set_xlabel('Frame')
        axs[1].set_ylabel('FLOPs')
        axs[1].legend()
        
        axs[2].plot(self.flops['pr'], label='Person Recognition', color='green')
        axs[2].set_title('FLOPs Over Frames')
        axs[2].set_xlabel('Frame')
        axs[2].set_ylabel('FLOPs')
        axs[2].legend()
        
        plt.tight_layout()
        
        plt.savefig(self.flops_path)
        print(f"[FrameToVideo] saved flops plot to {self.flops_path}")
        
        plt.close()

    def _end(self):
        self.draw_latency() # add
        self.draw_times() # add
        
        # self.draw_flops() # add
        
        self.end_flag = True
