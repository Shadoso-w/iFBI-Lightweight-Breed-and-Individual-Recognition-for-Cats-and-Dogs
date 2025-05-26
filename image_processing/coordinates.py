import os
import torch
from PIL import Image
from ultralytics import YOLO
import logging

# 禁用日志输出
logging.getLogger("ultralytics").setLevel(logging.WARNING)

# 设置设备
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 加载训练好的YOLOv8模型
yolo_model = YOLO('model/best_all.pt')
yolo_model.to(DEVICE)

# 定义数据路径和变量
path = '/home/xd/HUAWEI-CUP/petfinder_all'
output_path = 'all'  # 保存坐标的文件路径
os.makedirs(output_path, exist_ok=True)
coords_file_path = os.path.join(output_path, 'coordinates_petfinder_all.txt')

# 收集所有文件路径及其对应的检测坐标
coords = []


def select_highest_prob_box(boxes, img_size):
    if len(boxes) == 0:
        return [0, 0, img_size[0], img_size[1]]
    else:
        highest_prob_box = max(boxes, key=lambda x: x[1])[0]
        return [int(coord) for coord in highest_prob_box]


# 递归遍历子文件夹中的图片
for root, dirs, files in os.walk(path):
    for file in files:
        if file.endswith('.png'):
            img_path = os.path.join(root, file)
            print(f"image path: ", img_path)
            data = Image.open(img_path).convert('RGB')

            # 使用YOLOv8预测ROI
            # results = yolo_model(data)
            # face_boxes = results[0].boxes[0].xyxy.cpu().numpy()
            # body_boxes = results[0].boxes[1].xyxy.cpu().numpy()
            results = yolo_model(data)
            boxes = results[0].boxes.xyxy.cpu().numpy()
            class_ids = results[0].boxes.cls.cpu().numpy()
            scores = results[0].boxes.conf.cpu().numpy()
            face_boxes = [(boxes[i], scores[i]) for i in range(len(class_ids)) if class_ids[i] == 0]
            body_boxes = [(boxes[i], scores[i]) for i in range(len(class_ids)) if class_ids[i] == 1]

            # 选择置信度最高的面部与身体框
            face_box = select_highest_prob_box(face_boxes, data.size)
            body_box = select_highest_prob_box(body_boxes, data.size)
            
            # 获取大文件名和小文件名
            parent_dir = os.path.basename(root)

            # 收集文件路径及其对应的检测坐标
            if len(face_boxes) > 0:
                xmin_f, ymin_f, xmax_f, ymax_f = face_box
                xmin_b, ymin_b, xmax_b, ymax_b = body_box
                coords.append((parent_dir, file, xmin_f, ymin_f, xmax_f, ymax_f, xmin_b, ymin_b, xmax_b, ymax_b))
            else:
                # 没有面部检测框的情况下也记录图片路径，坐标设为-1
                xmin_b, ymin_b, xmax_b, ymax_b = body_box
                coords.append((parent_dir, file, -1, -1, -1, -1, xmin_b, ymin_b, xmax_b, ymax_b))
# 按路径排序
coords.sort()

# 写入坐标信息到文件
with open(coords_file_path, 'w') as coords_file:
    for coord in coords:
        parent_dir, file, xmin_f, ymin_f, xmax_f, ymax_f, xmin_b, ymin_b, xmax_b, ymax_b = coord
        coords_file.write(f"{parent_dir} {file} {xmin_f} {ymin_f} {xmax_f} {ymax_f} {xmin_b} {ymin_b} {xmax_b} {ymax_b}\n")

print(f'检测坐标已保存到: {coords_file_path}')