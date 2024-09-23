import os
import cv2
import time
import torch

import numpy as np

from queue import Empty
from multiprocessing import Process, Queue

from thop import profile
from configs import config
from request import Request

class LicenseRecognition(Process):
    def __init__(self, 
                 config: dict,
                 car_queue: Queue,
                 car_frame_queue: Queue):
        super().__init__()
        
        # from object_detection module
        self.car_queue = car_queue
        # to frame_to_video module
        self.car_frame_queue = car_frame_queue
        
        self.device = torch.device(config['license_recognition']['device'])
        self.model_path = config['license_recognition']['easyocr_model_path']
        
        self.monitor_interval = config['monitor_interval']
        
        self.model = None
        
        self.end_flag = False

    def run(self):
        print(f"[LicenseRecognition] Start!")
        
        self.model = torch.load(self.model_path)
        self.model.device = self.device
        
        while not self.end_flag:
            try:
                request = self.car_queue.get(timeout=1)
            except Empty:
                continue
            
            if request is None:
                self._end()
                break
            
            self._infer(request)
            
            # print(f"[LicenseRecognition] video_id: {request.video_id}, frame_id: {request.frame_id}, car_id: {request.car_id}")
            
    def _infer(self, request):
        flops, params = 0, 0
        
        if request.box is not None:
            frame_array = request.data
            
            frame_array = cv2.cvtColor(frame_array, cv2.COLOR_BGR2RGB)

            frame_size = frame_array.shape
            
            box = request.box
            
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
            
            inputs = frame_array[y1:y2, x1:x2]
            
            # try: # To measure the FLOPs and parameters of the model        
            #     inputs_array = np.array(inputs.copy().transpose(2, 0, 1), dtype=np.float32)
            #     inputs_tensor = torch.from_numpy(inputs_array).unsqueeze(0).to("cuda:0")
            #     flops, params = profile(self.model.detector.module, inputs=(inputs_tensor, )) # add，好像有点问题，在下面的 try 语句会报错
            # except Exception as e:
            #     pass

            with torch.no_grad():
                try: # add
                    outputs = self.model.readtext(inputs)
                except Exception as e:
                    print(f"[LicenseRecognition] Error: {e}")
                    print(f"[LicenseRecognition] inputs: {inputs.shape}")
                    outputs = []
                
            results = outputs
            
            label = "none"
            
            # select the highest confidence result
            if len(results) > 0:
                sorted_results = sorted(results, key=lambda x: x[2], reverse=True)
                label = sorted_results[0][1]

            request.label += f": {label}"
            
            # print(f"[LicenseRecognition] video_id: {request.video_id}, frame_id: {request.frame_id}, car_id: {request.car_id}, label: {request.label}")
        
        request.times.append(time.time())
        request.flops.append(flops)
        
        self.car_frame_queue.put(request)
        pass
            
    def _end(self):
        self.car_frame_queue.put(None)
        
        self.end_flag = True
        print(f"[LicenseRecognition] Stop!")
