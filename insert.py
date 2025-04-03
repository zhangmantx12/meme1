import cv2
import numpy as np
from tqdm import tqdm
import os  # 导入 os 模块

def add_text_to_image(image_path, detection_file, output_path, text="Hello, World!"):
    # 读取图片
    image = cv2.imread(image_path)

    # 读取检测结果
    with open(detection_file, "r") as f:
        detections = [list(map(int, line.strip().split(", "))) for line in f.readlines()]

    # 定义字体和颜色
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    font_color = (255, 255, 255)
    thickness = 2

    # 在每个检测框的中心位置添加文字
    for box in detections:
        x1, y1, x2, y2 = box
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2

        # 计算文字的宽度和高度
        (text_width, text_height), _ = cv2.getTextSize(text, font, font_scale, thickness)

        # 计算文字的起始位置，使其居中
        text_x = center_x - text_width // 2
        text_y = center_y + text_height // 2

        # 在图片上添加文字
        cv2.putText(image, text, (text_x, text_y), font, font_scale, font_color, thickness)

    # 保存图片
    cv2.imwrite(output_path, image)

# 图片文件夹路径
image_folder = "image"
output_folder = "output_image"

# 创建输出文件夹
os.makedirs(output_folder, exist_ok=True)

# 获取所有图片文件
image_files = [f for f in os.listdir(image_folder) if f.endswith(('.png', '.jpg', '.jpeg'))]

# 处理所有图片
for image_file in tqdm(image_files, desc="Adding text to images"):
    image_path = os.path.join(image_folder, image_file)
    detection_file = os.path.join(output_folder, f"{os.path.splitext(image_file)[0]}_detections.txt")
    output_path = os.path.join(output_folder, image_file)
    add_text_to_image(image_path, detection_file, output_path)