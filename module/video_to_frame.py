import os
import cv2
import time
import torch

import numpy as np

from queue import Empty
from multiprocessing import Process, Queue

from configs import config
from request import Request

class VideoToFrame(Process):
    def __init__(self, 
                 config: dict,
                 frame_queue: Queue):
        # super(VideoToFrame, self).__init__()
        super().__init__()
        
        self.input_video_dir = config['input_video_dir']
        self.video_number = config['video_number']
        self.video_start_id = config['video_start_id']
        self.frame_interval = config['frame_interval']
        self.frame_size = config['frame_size']
        
        self.input_image_dir = config['input_image_dir']
        
        # to object_detection module
        self.frame_queue = frame_queue
        
        self.end_flag = False

    def run(self):
        print(f"[VideoToFrame] Start!")
        
        self._read_image()
        
        self._end()
        
    def _read_image(self):
        input_image_files = os.listdir(self.input_image_dir)
        # sort by image name
        input_image_files.sort(key=lambda x: int(x.split('.')[0])) # input_image_files ?
        
        image_files = sorted([f for f in os.listdir(self.input_image_dir) if f.endswith('.png')])
        
        video_id = 0
        frame_id = 0
        total_frames = len(image_files)
        video_fps = 30
        
        print(f"[VideoToFrame] video_id: {video_id}, total_frames: {total_frames}, video_fps: {video_fps}")
        
        adjust_frame_interval = time.time()

        for image_name in image_files:
            image_path = os.path.join(self.input_image_dir, image_name)

            # 读取图片
            image = cv2.imread(image_path)
            
            image_array = np.array(image)
            
            request = Request(
                video_id=video_id,
                frame_id=frame_id,
                frame_number=total_frames,
                
                video_fps=video_fps,
                data=image_array,
                start_time=time.time(),
                times=[],
                flops=[],
            )
            frame_id += 1
            
            self.frame_queue.put(request)
            
            time.sleep(max(0, self.frame_interval / 1000 - (time.time() - adjust_frame_interval)))
            adjust_frame_interval = time.time()
            
        
    def _read_video(self):
        input_video_files = os.listdir(self.input_video_dir)
        # sort by video name
        input_video_files.sort(key=lambda x: int(x.split('.')[0]))
        print(f"[VideoToFrame] input_video_files: {input_video_files}")
        
        for video_id in range(self.video_start_id, self.video_number + self.video_start_id):
            input_video_path = os.path.join(self.input_video_dir, f"{video_id % len(input_video_files)}.mp4")
            print(f"[VideoToFrame] input_video_path: {input_video_path}")
            
            cap = cv2.VideoCapture(input_video_path)
            video_fps = int(cap.get(5))
            
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            print(f"[VideoToFrame] video_id: {video_id}, total_frames: {total_frames}, video_fps: {video_fps}")
            
            frame_id = 0
            
            adjust_frame_interval = time.time()
            while True:
                ret, frame = cap.read() # frame: (height, width, channel), BGR
                if not ret:
                    break
                # frame.resize((224, 224, 3))
                frame_array = np.array(frame) # , dtype=np.float32) # .transpose(2, 0, 1)
                # print(f"[VideoToFrame] frame_array.shape = {frame_array.shape}")
                
                request = Request(
                    video_id=video_id,
                    frame_id=frame_id,
                    frame_number=total_frames,
                    
                    video_fps=video_fps,
                    data=frame_array,
                    start_time=time.time(),
                )
                frame_id += 1
                
                self.frame_queue.put(request)
                
                time.sleep(max(0, self.frame_interval / 1000 - (time.time() - adjust_frame_interval)))
                adjust_frame_interval = time.time()
                
            cap.release()
        
    def _end(self):
        self.frame_queue.put(None)
        
        self.end_flag = True
        print(f"[VideoToFrame] Stop!")
        
        
if __name__ == '__main__':
    frame_queue = Queue()
    video_to_frame = VideoToFrame(config, frame_queue)
    video_to_frame.start()
    try:
        video_to_frame.join()
    except KeyboardInterrupt:
        print("[main] KeyboardInterrupt")
        video_to_frame.terminate()
    print("[main] TEST END")
