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
                 person_frame_queue: Queue):
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
        self.car_number = {}
        self.person_number = {}

        self.end_flag = False

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
            
            if request.box is not None:
                self.video_dict[video_id][frame_id]['car'][car_id] = {'box': request.box, 'label': request.label}
            
            if len(self.video_dict[video_id]) == request.frame_number:
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
            
            if request.box is not None:
                self.video_dict[video_id][frame_id]['person'][person_id] = {'box': request.box, 'label': request.label}

            if len(self.video_dict[video_id]) == request.frame_number:
                self.check_video(video_id)
    
    # Check if the video contains all the frames, and if the frames contain all the cars and persons
    # If so, save the video as a video file
    def check_video(self, video_id):
        # print(f"[FrameToVideo] Check video: {video_id}")
        frame_number = self.video_dict[video_id][0]['request'].frame_number
        
        for frame_id in range(frame_number):
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
                x1 = int(x1 * frame_size[1])
                y1 = int(y1 * frame_size[0])
                x2 = int(x2 * frame_size[1])
                y2 = int(y2 * frame_size[0])
                
                # print(f"[FrameToVideo] Car box: ({x1}, {y1}, {x2}, {y2}) label {label}")
                
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(frame, label, (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            for person_id, person in self.video_dict[video_id][frame_id]['person'].items():
                box = person['box']
                label = person['label']
                
                # Relative coordinates need to be converted to absolute coordinates
                x1, y1, x2, y2 = box
                x1 = int(x1 * frame_size[1])
                y1 = int(y1 * frame_size[0])
                x2 = int(x2 * frame_size[1])
                y2 = int(y2 * frame_size[0])
                
                # print(f"[FrameToVideo] Person box: ({x1}, {y1}, {x2}, {y2}) label {label}")
                
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, label, (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            video.write(frame)
            
        video.release()
        
        # print(f"[FrameToVideo] Save video: {video_id} done! video_dict.keys(): {self.video_dict.keys()}")
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
        process_time_list = [self.process_time_dict[frame_id] for frame_id in range(len(self.process_time_dict))]
        car_number_list = [self.car_number[frame_id] for frame_id in range(len(self.car_number))]
        person_number_list = [self.person_number[frame_id] for frame_id in range(len(self.person_number))]
        
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
        plt.savefig('../latency/3.png')
    
        # plt.plot(range(len(process_time_list)), process_time_list)
        # plt.savefig('../latency/2.pdf')

    def _end(self):
        self.draw_latency() # add
        
        self.end_flag = True
