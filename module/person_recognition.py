import os
import cv2
import time
import torch
import pickle
import face_recognition

import numpy as np

from queue import Empty
from torchvision.models import resnet50
from multiprocessing import Process, Queue

from thop import profile
from configs import config
from request import Request

class PersonRecognition(Process):
    def __init__(self, 
                 config: dict,
                 person_queue: Queue,
                 person_frame_queue: Queue):
        super().__init__()
        
        # from object_detection module
        self.person_queue = person_queue
        # to frame_to_video module
        self.person_frame_queue = person_frame_queue
        
        self.device = torch.device(config['person_recognition']['device'])
        self.model_path = config['person_recognition']['face_recognition_model_path']
        
        self.monitor_interval = config['monitor_interval']
        
        self.model = None
        self.known_face_encodings = None
        self.known_face_names = None
        
        self.end_flag = False

    def run(self):
        print(f"[PersonRecognition] Start!")
        
        # Read the face recognition model
        with open(self.model_path, "rb") as f:
            self.model = resnet50().to("cuda:0") # add
            self.known_face_encodings = pickle.load(f)
            self.known_face_names = pickle.load(f)
        
        while not self.end_flag:
            try:
                request = self.person_queue.get(timeout=1)
            except Empty:
                continue
            
            if request is None:
                self._end()
                break
            
            self._infer(request)
            
            # print(f"[PersonRecognition] video_id: {request.video_id}, frame_id: {request.frame_id}, person_id: {request.person_id}")
            
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
            
            inputs = np.ascontiguousarray(inputs) # contiguous memory
            
            # try: # To measure the FLOPs and parameters of the model
            #     inputs_array = np.array(inputs.copy().transpose(2, 0, 1), dtype=np.float32)
            #     inputs_tensor = torch.from_numpy(inputs_array).unsqueeze(0).to("cuda:0")
            #     flops, params = profile(self.model, inputs=(inputs_tensor, )) # add，好像有点问题，在下面的 try 语句会报错
            # except Exception as e:
            #     pass
            
            # print(f"[PersonRecognition] inputs.shape = {inputs.shape}")
            
            with torch.no_grad():
                try:
                    face_encodings = face_recognition.face_encodings(inputs)
                except Exception as e:
                    print(f"[PersonRecognition] Error: face_recognition.face_encodings error: {e}")
                    face_encodings = []
            
            label = "none"
            
            if len(face_encodings) > 0:
                matches = face_recognition.compare_faces(self.known_face_encodings, face_encodings[0])
                name = "unknown"
                
                if True in matches:
                    first_match_index = matches.index(True)
                    name = self.known_face_names[first_match_index]
                
                label = name
                
            request.label += f": {label}"
        
        request.times.append(time.time())
        request.flops.append(flops)
        
        self.person_frame_queue.put(request)
        pass
            
    def _end(self):
        self.person_frame_queue.put(None)
        
        self.end_flag = True
        print(f"[PersonRecognition] Stop!")
