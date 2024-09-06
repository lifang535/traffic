import os
import cv2
import numpy as np

# 自定义高斯噪声函数
def gaussian_noise(image, mean, var):
    image = np.array(image / 255, dtype=float)
    noise = np.random.normal(mean, var ** 0.5, image.shape)
    out = image + noise

    if out.min() < 0:
        low_clip = -1.
    else:
        low_clip = 0.

    out = np.clip(out, low_clip, 1.0)
    out = np.uint8(out * 255)

    return out

input_image_dir = "/root/lifang535/nsl_project/efficiency_attack/multi-model_application/traffic/input_image/after_attacking"
output_image_dir = "/root/lifang535/nsl_project/efficiency_attack/multi-model_application/traffic/input_image/after_defending"

# 确保输出目录存在
os.makedirs(output_image_dir, exist_ok=True)

# 遍历输入目录中的所有文件
for filename in os.listdir(input_image_dir):
    if filename.endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')):
        # 构建完整的文件路径
        input_image_path = os.path.join(input_image_dir, filename)
        
        # 读取图像
        image = cv2.imread(input_image_path)
        
        # 添加高斯噪声
        noisy_image = gaussian_noise(image, mean=0, var=0.01)  # 根据需要调整mean和var
        
        # 构建输出文件路径
        output_image_path = os.path.join(output_image_dir, filename)
        
        # 保存图像
        cv2.imwrite(output_image_path, noisy_image)

print("所有图像已处理并保存到输出目录。")
