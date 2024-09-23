from collections import defaultdict
from loguru import logger
from tqdm import tqdm

import torch

# from yolox.utils import (
#     gather,
#     is_main_process,
#     postprocess,
#     synchronize,
#     time_synchronized,
#     xyxy2xywh
# )

import contextlib
import io
import os
import itertools
import json
import tempfile
import time
import numpy as np
import torch.nn.functional as F
from torch.optim import Adam
import cv2

import sys
import torchvision

from pathlib import Path

YOLOV5_FILE = Path(f"../model/yolov5").resolve()
if str(YOLOV5_FILE) not in sys.path:
    sys.path.append(str(YOLOV5_FILE))  # add ROOT to PATH
from models.common import DetectMultiBackend
from utils.general import Profile, non_max_suppression

from PIL import Image
import logging

def create_logger(module, filename, level):
    # Create a formatter for the logger, setting the format of the log with time, level, and message
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    
    # Create a logger named 'logger_{module}'
    logger = logging.getLogger(f'logger_{module}')
    logger.setLevel(level)     # Set the log level for the logger
    
    # Create a file handler, setting the file to write the logs, level, and formatter
    fh = logging.FileHandler(filename, mode='w')
    fh.setLevel(level)         # Set the log level for the file handler
    fh.setFormatter(formatter) # Set the formatter for the file handler
    
    # Add the file handler to the logger
    logger.addHandler(fh)
    
    return logger

def generate_mask(outputs, x_shape, y_shape): # 第一次推理后生成 mask
    mask_x = 4
    mask_y = 2
    mask = torch.ones(y_shape,x_shape)
    
    conf_thres = 0.0 # confidence threshold # lifang535
    iou_thres = 0.0  # NMS IOU threshold
    max_det = 1000   # maximum detections per image
    outputs = non_max_suppression(prediction=outputs, conf_thres=conf_thres, iou_thres=iou_thres, max_det=max_det)
    outputs = outputs[0]
    
    x_len = int(x_shape / mask_x)
    y_len = int(y_shape / mask_y)
    if outputs is not None:
        for i in range(len(outputs)):
            detection = outputs[i]
            center_x, center_y = (detection[0]+detection[2])/2, (detection[1]+detection[3])/2
            # 根据检测框的中心点位置，判断它在哪个区域
            region_x = int(center_x / x_len)
            region_y = int(center_y / y_len)
            
            mask[region_y*y_len:(region_y+1)*y_len, region_x*x_len:(region_x+1)*y_len] -= 0.05
    
    # print(f"mask.shape = {mask.shape}")
    # print(f"mask = {mask}")
    
    return mask

def run_attack(outputs, bx, mask): # 每轮迭代的攻击
    outputs = outputs[0][0]
    scores = outputs[:, index] * outputs[:, 4] # class confidence * object confidence

    loss2 = 40 * torch.norm(bx, p=2) # 可能是为了让 bx 尽量小
    # targets = torch.ones_like(scores)
    targets = torch.zeros_like(scores)
    loss3 = F.mse_loss(scores, targets, reduction='sum')
    loss = loss3 # + loss2
    
    loss.requires_grad_(True)
    loss.backward(retain_graph=True)
    
    # adam_opt.step()
    bx.grad = bx.grad / (torch.norm(bx.grad, p=2) + 1e-20)
    bx.data = -3.5 * mask * bx.grad + bx.data
    count = (scores > 0.3).sum()
    print(f"loss: {loss.item()}: loss_2 {loss2.item()}: loss_3 {loss3.item()}: count: {count.item()}")
    return bx

def overload_attack(
    image_list,
    image_name_list,
):  
    global model, device
    
    def process_img(imgs, image_name):
        global model, names
        
        tensor_type = torch.cuda.FloatTensor
            
        frame_id = 0
        total_l1 = 0
        total_l2 = 0
        rgb_means=torch.tensor((0.485, 0.456, 0.406)).view(1, 3, 1, 1).to(device)
        std=torch.tensor((0.229, 0.224, 0.225)).view(1, 3, 1, 1).to(device)
        
        frame_id += 1
        bx = np.zeros((imgs.shape[1], imgs.shape[2], imgs.shape[3]))
        bx = bx.astype(np.float32)
        bx = torch.from_numpy(bx).to(device).unsqueeze(0)
        bx = bx.data.requires_grad_(True)
        imgs = imgs.type(tensor_type)
        imgs = imgs.to(device)
        #(1,23625,6)
        
        for iter in tqdm(range(epochs)):
            added_imgs = imgs+bx
            
            l2_norm = torch.sqrt(torch.mean(bx ** 2))
            l1_norm = torch.norm(bx, p=1)/(bx.shape[3]*bx.shape[2])
            added_imgs.clamp_(min=0, max=1)
            input_imgs = (added_imgs - rgb_means)/std
            outputs = model(input_imgs) # [torch.Size([1, 40698, 85])]

            if iter == 0:
                mask = generate_mask(outputs, added_imgs.shape[3], added_imgs.shape[2]).to(device)
            bx = run_attack(outputs, bx, mask)

        print(added_imgs.shape)
        added_blob = torch.clamp(added_imgs*255,0,255).squeeze().permute(1, 2, 0).detach().cpu().numpy()
        added_blob = added_blob[..., ::-1]
        # added_blob_2 = added_blob_2[..., ::-1]

        input_path = f"{input_dir}/{image_name}"
        output_path = f"{output_dir}/{image_name}"
        cv2.imwrite(output_path, added_blob)
        
        # output_path_tiff = output_path.replace(".png", ".tiff")
        # added_blob = torch.clamp(input_imgs*255,-255,510).squeeze().permute(1, 2, 0).detach().cpu().numpy()
        # added_blob = added_blob[..., ::-1]
        # cv2.imwrite(output_path_tiff, added_blob)
        
        print(f"saved image to {output_path}")
        objects_num_before_nms, objects_num_after_nms, person_num_after_nms, car_num_after_nms = infer(input_path)
        _objects_num_before_nms, _objects_num_after_nms, _person_num_after_nms, _car_num_after_nms = infer(output_path)
        
        logger.info(f"objects_num_before_nms: {objects_num_before_nms}, objects_num_after_nms: {objects_num_after_nms}, person_num_after_nms: {person_num_after_nms}, car_num_after_nms: {car_num_after_nms} -> _objects_num_before_nms: {_objects_num_before_nms}, _objects_num_after_nms: {_objects_num_after_nms}, _person_num_after_nms: {_person_num_after_nms}, _car_num_after_nms: {_car_num_after_nms}")
        # infer(output_path_tiff)
        
        print(l1_norm.item(),l2_norm.item())
        total_l1 += l1_norm
        total_l2 += l2_norm
        mean_l1 = total_l1/frame_id
        mean_l2 = total_l2/frame_id
        print(mean_l1.item(),mean_l2.item())
        del bx
        del outputs
        del imgs

        return mean_l1, mean_l2
        
    for image, image_name in zip(image_list, image_name_list):
        image = image.transpose((2, 0, 1))[::-1]
        image = np.ascontiguousarray(image)
        image = torch.from_numpy(image).to(device).float()
        image /= 255.0

        if len(image.shape) == 3:
            image = image[None]

        # print(f"image.shape = {image.shape}")
        
        mean_l1, mean_l2 = process_img(image, image_name)

    return

def infer(image_path):
    image = cv2.imread(image_path)

    image = image.transpose((2, 0, 1))[::-1]
    image = np.ascontiguousarray(image)
    image = torch.from_numpy(image).to(device).float()
    image /= 255.0
    
    if len(image.shape) == 3:
        image = image[None]
    
    # tensor_type = torch.cuda.FloatTensor
    # image_tensor = image.type(tensor_type)
    # image_tensor = image.to(device)
    
    image_tensor = image
    
    # print(f"image_tensor = {image_tensor}")
    
    outputs = model(image_tensor)
    
    # print(f"outputs = {outputs}")
    
    outputs = outputs[0].unsqueeze(0)
    
    # scores = outputs[..., index] * outputs[..., 4]
    # scores = scores[scores > 0.25]
    # print(f"len(scores) = {len(scores)}")
    # objects_num_before_nms = len(scores) # 实际上是 {attack_object} number before NMS
    
    conf_thres = 0.25 # 0.25  # confidence threshold
    iou_thres = 0.45  # 0.45  # NMS IOU threshold
    max_det = 1000    # maximum detections per image
    
    xc = outputs[..., 4] > 0
    x = outputs[0][xc[0]]
    x[:, 5:] *= x[:, 4:5]
    max_scores = x[:, 5:].max(dim=-1).values
    objects_num_before_nms = len(max_scores[max_scores > 0.25]) # 这个是对的，用最大的 class confidence 筛选
    
    objects_num_after_nms = 0
    person_num_after_nms = 0
    car_num_after_nms = 0
    
    outputs = non_max_suppression(outputs, conf_thres, iou_thres, max_det=max_det)
    
    for i, det in enumerate(outputs): # detections per image
        if len(det):
            for *xyxy, conf, cls in reversed(det):
                c = int(cls)
                label = f"{names[c]}"
                confidence = float(conf)
                confidence_str = f"{confidence}" # f"{confidence:.2f}"
                box = [round(float(i), 2) for i in xyxy]
                # print(f"Detected {label} with confidence {confidence_str} at location {box}")
                if label == "person":
                    person_num_after_nms += 1
                elif label == "car":
                    car_num_after_nms += 1
            objects_num_after_nms = len(det)
        # print(f"There are {len(det)} objects detected in this image.")
    
    # objects_num_before_nms, objects_num_after_nms, person_num_after_nms, car_num_after_nms
    print(f"objects_num_before_nms = {objects_num_before_nms}, objects_num_after_nms = {objects_num_after_nms}, person_num_after_nms = {person_num_after_nms}, car_num_after_nms = {car_num_after_nms}")
    return objects_num_before_nms, objects_num_after_nms, person_num_after_nms, car_num_after_nms
        
def dir_process(dir_path):
    image_list = []
    image_name_list = os.listdir(dir_path)
    image_name_list.sort()
    # print(f"image_name_list = {image_name_list}")
    for image_name in image_name_list:
        if image_name.endswith(".png"):
            image_path = os.path.join(dir_path, image_name)
            image = cv2.imread(image_path)
            image_list.append(image)

    return image_list, image_name_list

if __name__ == "__main__":
    # image_name = "000001.png"
    # input_path = f"original_image/{image_name}"
    # output_path = f"overload_attack_image/{image_name}"
    # infer(output_path)

    weights = "../model/yolov5/yolov5n.pt" # yolov5s.pt yolov5m.pt yolov5l.pt yolov5x.pt
    device = torch.device('cuda:3')
    model = DetectMultiBackend(weights=weights, device=device)
    names = model.names
    print(f"names = {names}")
    
    attack_object_key = 0 # 0: person, 2: car
    attack_object = names[attack_object_key]
    index = 5 + attack_object_key # yolov5 输出的结果中，class confidence 对应的 index

    epochs = 200
    
    logger = create_logger(f"overload_attack_{attack_object}_epochs_{epochs}", f"overload_attack_{attack_object}_epochs_{epochs}.log", logging.INFO)
    
    input_dir = "original_image"
    output_dir = f"overload_attack_image/{attack_object}_epochs_{epochs}"
    
    start_time = time.time()
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    image_list, image_name_list = dir_process(input_dir)
    overload_attack(image_list, image_name_list)
    
    # TODO: 测一下哪步时延长
