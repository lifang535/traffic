from collections import defaultdict
from loguru import logger
from tqdm import tqdm

import torch

from yolox.utils import (
    gather,
    is_main_process,
    postprocess,
    synchronize,
    time_synchronized,
    xyxy2xywh
)

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
learning_rate = 0.02 #0.07
epochs = 500

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

yolo_model_path = f"../model/yolo-tiny/yolos-tiny_model.pth"
image_processor_path = f"../model/yolo-tiny/yolos-tiny_image_processor.pth"
yolo_model = torch.load(yolo_model_path, map_location=device)
yolo_image_processor = torch.load(image_processor_path, map_location=device)

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

logger = create_logger("attack_data", "attack_data.log", logging.INFO)

def generate_mask(outputs,result, x_shape, y_shape):

    mask_x = 4
    mask_y = 2
    # mask = torch.ones(mask_y,mask_x)  # 初始mask为3*3
    mask = torch.ones(y_shape,x_shape)
    
    boxes = result["boxes"]

    x_len = int(x_shape / mask_x)
    y_len = int(y_shape / mask_y)
    if boxes is not None:
        for i in range(len(boxes)):
            detection = boxes[i]
            center_x, center_y = (detection[0]+detection[2])/2, (detection[1]+detection[3])/2

            # Based on the position of the center of the detection box, determine which region it is in
            region_x = int(center_x / x_len)
            region_y = int(center_y / y_len)
            
            mask[region_y*y_len:(region_y+1)*y_len, region_x*x_len:(region_x+1)*y_len] -= 0.05
    
    
    return mask

def run_attack(outputs,result,bx, strategy, max_tracker_num, mask):

    per_num_b = (25*45)/max_tracker_num
    per_num_m = (50*90)/max_tracker_num
    per_num_s = (100*180)/max_tracker_num

    # scores = outputs[:,5] * outputs[:,4] # remove
    
    scores = result["scores"] # add

    loss2 = 40*torch.norm(bx, p=2)
    targets = torch.ones_like(scores)
    loss3 = F.mse_loss(scores, targets, reduction='sum')
    loss = loss3#+loss2
    
    loss.requires_grad_(True)
    loss.backward(retain_graph=True)
    
    bx.grad = bx.grad / (torch.norm(bx.grad,p=2) + 1e-20)
    bx.data = -3.5 * mask * bx.grad+ bx.data
    count = (scores > 0.9).sum()
    print('loss',loss.item(),'loss_2',loss2.item(),'loss_3',loss3.item(),'count:',count.item())
    return bx

def attack(
    frame_list,
    half=False,
):
    tensor_type = torch.cuda.HalfTensor if half else torch.cuda.FloatTensor
        
    frame_id = 0
    total_l1 = 0
    total_l2 = 0
    strategy = 0
    max_tracker_num = int(15)
    rgb_means=torch.tensor((0.485, 0.456, 0.406)).view(1, 3, 1, 1).to(device)
    std=torch.tensor((0.229, 0.224, 0.225)).view(1, 3, 1, 1).to(device)
    
    def process_img(imgs):
        nonlocal frame_id, total_l1, total_l2, strategy, max_tracker_num, rgb_means, std

        frame_id += 1
        bx = np.zeros((imgs.shape[1], imgs.shape[2], imgs.shape[3]))
        bx = bx.astype(np.float32)
        bx = torch.from_numpy(bx).to(device).unsqueeze(0)
        bx = bx.data.requires_grad_(True)
        # imgs = imgs.type(tensor_type)
        imgs = imgs.to(device)
        
        for iter in tqdm(range(epochs)):
            # print(f"========== iter: {iter} ==========")
            
            added_imgs = imgs+bx
            
            l2_norm = torch.sqrt(torch.mean(bx ** 2))
            l1_norm = torch.norm(bx, p=1)/(bx.shape[3]*bx.shape[2])
            
            outputs = None

            yolo_outputs = yolo_model(added_imgs)

            target_sizes = [imgs.shape[2:] for _ in range(1)]
            process_yolo_outputs = yolo_image_processor.post_process_object_detection(yolo_outputs, threshold=0.0, target_sizes=target_sizes)
            result = process_yolo_outputs[0]
            
            if iter == 0:
                mask = generate_mask(outputs,result,added_imgs.shape[3],added_imgs.shape[2]).to(device) # The mask is generated only once
            bx = run_attack(outputs,result,bx, strategy, max_tracker_num, mask)

        if strategy == max_tracker_num-1:
            strategy = 0
        else:
            strategy += 1
        print(added_imgs.shape)
        added_blob = torch.clamp(added_imgs*255,0,255).squeeze().permute(1, 2, 0).detach().cpu().numpy()
        added_blob = added_blob[..., ::-1]
        
        # print(f"max: {added_blob.max()}, min: {added_blob.min()}")
        added_blob = np.round(added_blob).astype(np.uint8)
        
        # frame_id = 0 # TODO
        save_path = f"{attacking_image_dir}/{frame_id:06d}.png"
        # print(f"========== save_path: {save_path} ==========")
        cv2.imwrite(save_path, added_blob)
        
        # Image.fromarray(added_blob).save(save_path)
        
        # See the result of the attacking image
        test_img = Image.open(save_path)
        test_img = np.array(test_img).transpose(2, 0, 1)
        test_img = (torch.from_numpy(test_img).unsqueeze(0).float() / 255.0).to(device)
        yolo_outputs_read = yolo_model(test_img)
        process_yolo_outputs_read = yolo_image_processor.post_process_object_detection(yolo_outputs_read, threshold=0.9, target_sizes=target_sizes)
        result_read = process_yolo_outputs_read[0]
        
        # See the result of the original image
        yolo_outputs_original = yolo_model(imgs)
        process_yolo_outputs_original = yolo_image_processor.post_process_object_detection(yolo_outputs_original, threshold=0.9, target_sizes=target_sizes)
        result_original = process_yolo_outputs_original[0]
        
        # See the result of the attacking image (float)
        yolo_outputs_float = yolo_model(added_imgs)
        process_yolo_outputs_float = yolo_image_processor.post_process_object_detection(yolo_outputs_float, threshold=0.9, target_sizes=target_sizes)
        result_float = process_yolo_outputs_float[0]
        
        # See the result of the attacking image (int)
        added_imgs_int = torch.round((added_imgs * 255)).to(torch.uint8) # .float() / 255.0
        added_imgs_int = (added_imgs_int.float() / 255.0)
        yolo_outputs_int = yolo_model(added_imgs_int)
        process_yolo_outputs_int = yolo_image_processor.post_process_object_detection(yolo_outputs_int, threshold=0.9, target_sizes=target_sizes)
        result_int = process_yolo_outputs_int[0]
        
        print(f"[Frame {frame_id}] The number of detected objects: {len(result_original['labels'])} → {len(result_read['labels'])} (if float: {len(result_float['labels'])}, if int: {len(result_int['labels'])})")
        logger.info(f"[Frame {frame_id}] The number of detected objects: {len(result_original['labels'])} → {len(result_read['labels'])} (if float: {len(result_float['labels'])}, if int: {len(result_int['labels'])})")
        # for score, label, box in zip(result["scores"], result["labels"], result["boxes"]):
        #     box = [round(i, 2) for i in box.tolist()]
        #     print(
        #         f"Detected {yolo_model.config.id2label[label.item()]} with confidence "
        #         f"{round(score.item(), 3)} at location {box}"
        #     )
        
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
        
    for frame_id, frame in enumerate(frame_list):
        mean_l1, mean_l2 = process_img(frame)

    return mean_l1, mean_l2


if __name__ == "__main__":
    # # Read the video and extract the frames
    # # Input: Video
    # # Output: Attacking images
    # frame_list = []
    
    # input_video_path = f"0.mp4"
    # cap = cv2.VideoCapture(input_video_path)
    
    # video_fps = int(cap.get(5))
    # total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    # print(f"video_fps = {video_fps}, total_frames = {total_frames}")
    
    # while True:
    #     ret, frame = cap.read() # frame: (height, width, channel), BGR
    #     if not ret:
    #         break
    #     # frame.resize((224, 224, 3))
        
    #     frame_array = np.array(frame) # , dtype=np.float32) # .transpose(2, 0, 1)
    #     # print(f"frame_array.shape = {frame_array.shape}")
        
    #     cv2.imwrite(f"../input_image/before_attacking/{(len(frame_list) + 1):06d}.png", frame_array) # jpg -> png
    #     frame = cv2.imread(f"../input_image/before_attacking/{(len(frame_list) + 1):06d}.png")
    #     frame_array = np.array(frame)
        
    #     frame_array = cv2.cvtColor(frame_array, cv2.COLOR_BGR2RGB)
        
    #     frame_tensor = torch.tensor(frame_array).permute(2, 0, 1).float() / 255.0
    #     frame_tensor = frame_tensor.unsqueeze(0)
        
    #     # print(f"frame_tensor.shape = {frame_tensor.shape}")
        
    #     frame_list.append(frame_tensor)
        
    # cap.release()
    
    # attacking_image_dir = f"../input_image/after_attacking"
    
    # attack(frame_list=frame_list)
    
    
    
    # Read the video and extract the frames
    # Input: Original images
    # Output: Attacking images
    image_list = []
    
    original_image_dir = f"../input_image/before_attacking"
    attacking_image_dir = f"../input_image/after_attacking"

    # sort by image name
    original_image_files = sorted([f for f in os.listdir(original_image_dir) if f.endswith('.png')])
    
    for original_image_file in original_image_files:
        original_image_path = f"{original_image_dir}/{original_image_file}"
        
        image = cv2.imread(original_image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = torch.tensor(image).permute(2, 0, 1).float() / 255.0
        image = image.unsqueeze(0)
        
        image_list.append(image)
        
    attack(frame_list=image_list)
    
    
    

