import os
import sys
import cv2
import time
import torch

import numpy as np

from PIL import Image
from queue import Empty
from threading import Thread
from multiprocessing import Process, Queue
from concurrent.futures import ThreadPoolExecutor

from thop import profile
from configs import config
from request import Request

from pathlib import Path
YOLOV5_FILE = Path(f"../model/yolov5").resolve()
if str(YOLOV5_FILE) not in sys.path:
    sys.path.append(str(YOLOV5_FILE))  # add YOLOV5_FILE to PATH
from models.common import DetectMultiBackend
from utils.general import Profile, non_max_suppression

class ObjectDetection(Process):
    def __init__(self, 
                 config: dict,
                 frame_queue: Queue,
                 car_queue: Queue,
                 person_queue: Queue,):
        super().__init__()

        # from video_to_frame module
        self.frame_queue = frame_queue
        # to license_recognition module
        self.car_queue = car_queue
        # to person_recognition module
        self.person_queue = person_queue
        
        self.frame_size = config['frame_size']
        
        # self.device = torch.device(config['object_detection']['device']) # lifang535 remove
        # self.image_processor_path = config['object_detection']['yolo-tiny_image_processor_path']
        # self.model_path = config['object_detection']['yolo-tiny_model_path']
        
        self.device = torch.device(config['yolov5']['device']) # lifang535 add
        self.yolov5n_weights_path = config['yolov5']['yolov5n_weights_path']
        self.conf_thres = config['yolov5']['conf_thres']
        self.iou_thres = config['yolov5']['iou_thres']
        self.max_det = config['yolov5']['max_det']
        self.target_size = None
        
        # self.monitor_interval = config['monitor_interval'] # lifang535
        
        self.image_processor = None
        self.model = None
        self.id2label = None
        
        # self.thread_pool = ThreadPoolExecutor(max_workers=1000)
        
        self.end_flag = False

    def run(self):
        print(f"[ObjectDetection] Start!")
        
        # self.image_processor = torch.load(self.image_processor_path, map_location=self.device) # lifang535 remove
        # self.model = torch.load(self.model_path, map_location=self.device)
        # self.id2label = self.model.config.id2label
        
        self.model = DetectMultiBackend(weights=self.yolov5n_weights_path, device=self.device) # lifang535 add
        self.id2label = self.model.names
        # image_size = (480, 640)
        # self.model.warmup(imgsz=(1, 3, *image_size))  # warmup
        
        while not self.end_flag:
            try:
                request = self.frame_queue.get(timeout=1)
            except Empty:
                continue
            
            if request is None:
                self._end()
                break
            
            # temp_thread = self.thread_pool.submit(self._infer, request)
            
            self._infer(request)
            
            if request.frame_id % 50 == 0:
                print(f"[ObjectDetection] video_id: {request.video_id}, frame_id: {request.frame_id}")
    
    def _infer(self, request):
        def _preprocess(image_array): # lifang535 add
            image_array = image_array.transpose((2, 0, 1))[::-1]
            image_array = np.ascontiguousarray(image_array)
            image_tensor = torch.from_numpy(image_array).to(self.device).float()
            image_tensor /= 255.0
            if len(image_tensor.shape) == 3:
                image_tensor = image_tensor[None]
            
            # print(f"[ObjectDetection] image_tensor = {image_tensor}")
            
            return image_tensor
        
        def _postprocess(outputs): # with batch size 1 # lifang535 add
            outputs = outputs[0].unsqueeze(0)
            outputs = non_max_suppression(prediction=outputs, conf_thres=self.conf_thres, iou_thres=self.iou_thres, max_det=self.max_det)
            results = []
            for output in outputs:
                result = {
                    "boxes": [],
                    "scores": [],
                    "labels": [],
                }
                # print(f"[ObjectDetection] len(output) = {len(output)}")
                # time.sleep(1000000)
                if len(output):
                    for *xyxy, conf, cls in reversed(output):
                        c = int(cls)
                        label = f"{self.id2label[c]}"
                        confidence = float(conf)
                        box = [float(i) for i in xyxy] # TODO: 0 ~ 1
                        result["boxes"].append(box)
                        result["scores"].append(confidence)
                        result["labels"].append(label)
                results.append(result)
            return results
        
        flops, params = 0, 0
        
        frame_array = request.data
        
        # frame_array = cv2.cvtColor(frame_array, cv2.COLOR_BGR2RGB) # lifang535 remove
        # # inputs = self.image_processor(images=[frame_array], return_tensors="pt").to(self.device)
        # inputs = {
        #     'pixel_values': (torch.from_numpy(frame_array.transpose(2, 0, 1)).unsqueeze(0).float() / 255.0).to(self.device)
        # }
        
        inputs = _preprocess(frame_array) # lifang535 add
        
        with torch.no_grad():
            # outputs = self.model(**inputs)
            # outputs = self.model(inputs['pixel_values']) # lifang535 remove
            outputs = self.model(inputs)

        # results = self.image_processor.post_process_object_detection(outputs, threshold=0.9) # lifang535 remove
        
        # try: # To measure the FLOPs and parameters of the model
        #     flops, params = profile(self.model, inputs=(inputs['pixel_values'], )) # add
        # except Exception as e:
        #     pass
        
        # if request.frame_id % 1 == 0: # lifang535 remove
        #     print(f"[ObjectDetection] frame_array.shape = {frame_array.shape}, inputs['pixel_values'].shape = {inputs['pixel_values'].shape}")
        #     print(f"[ObjectDetection] video_id: {request.video_id}, frame_id: {request.frame_id}, len(results[0]['labels']) = {len(results[0]['labels'])}")
        
        # print(f"[ObjectDetection] Inference time: {round(time.time() - start_time, 4)}")
        
        results = _postprocess(outputs) # lifang535 add
        
        for i, result in enumerate(results): # lifang535 remove
            # car_number = sum([1 for label in result["labels"] if self.id2label[label.item()] == 'car']) # lifang535 remove
            # person_number = sum([1 for label in result["labels"] if self.id2label[label.item()] == 'person']) # lifang535 remove
            car_number = sum([1 for label in result["labels"] if label == 'car']) # lifang535 add
            person_number = sum([1 for label in result["labels"] if label == 'person']) # lifang535 add
            
            # print(f"[ObjectDetection] car_number = {car_number}, person_number = {person_number}")
            
            request.car_number = car_number
            request.person_number = person_number
            request.times.append(time.time())
            request.flops.append(flops)
            
            self.car_queue.put(request)
            self.person_queue.put(request)
            
            car_id = 0
            person_id = 0
            
            for score, label, box in zip(result["scores"], result["labels"], result["boxes"]):
                # box = [round(i, 5) for i in box.tolist()] # lifang535 remove
                box = [round(i, 5) for i in box] # lifang535 add
                # if self.id2label[label.item()] == 'car': # lifang535 remove
                if label == 'car': # lifang535 add
                    req = request.copy()
                    req.car_id = car_id
                    
                    req.box = box
                    # req.label = self.id2label[label.item()] # lifang535 remove
                    req.label = label # lifang535 add
                    self.car_queue.put(req)
                    
                    car_id += 1
                # elif self.id2label[label.item()] == 'person': # lifang535 remove
                elif label == 'person': # lifang535 add
                    req = request.copy()
                    req.person_id = person_id
                    
                    req.box = box
                    # req.label = self.id2label[label.item()] # lifang535 remove
                    req.label = label # lifang535 add
                    self.person_queue.put(req)
                    
                    person_id += 1
                    
                # print(
                #     f"Detected {self.id2label[label.item()]} with confidence "
                #     f"{round(score.item(), 3)} at location {box}"
                # )
                    
    # def _monitor(self): # lifang535
    #     qsizes = []
        
    #     adjust_monitor_interval = time.time()
    #     while not self.end_flag:
    #         qsizes.append(self.frame_queue.qsize())
            
    #         time.sleep(max(0, self.monitor_interval / 1000 - (time.time() - adjust_monitor_interval)))
    #         adjust_monitor_interval = time.time()
            
    #     # draw queue size
        
            
            
            
            
            
    def _end(self):
        self.car_queue.put(None)
        self.person_queue.put(None)
        
        self.end_flag = True
        print(f"[ObjectDetection] Stop!")

        
if __name__ == '__main__':
    frame_queue = Queue()
    object_detection = ObjectDetection(config, frame_queue)
    object_detection.start()
    try:
        object_detection.join()
    except KeyboardInterrupt:
        print("[main] KeyboardInterrupt")
        object_detection.terminate()
        