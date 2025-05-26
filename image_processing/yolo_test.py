import os
import torch
from PIL import Image
import logging
from ultralytics import YOLO

def yolo_test(image_path):
    logging.getLogger("ultralytics").setLevel(logging.WARNING)
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    yolo_model = YOLO('model/best.pt')
    yolo_model.to(DEVICE)           

    data = Image.open(image_path).convert('RGB')

    # 使用YOLOv8预测ROI
    results = yolo_model(data)
    boxes = results[0].boxes.xyxy.cpu().numpy()

    if len(boxes) > 0:
        xmin, ymin, xmax, ymax = boxes[0]
    else:
        # 没有检测框的情况下也记录图片路径，坐标设为-1
        xmin, ymin, xmax, ymax = -1, -1, -1, -1

    bbox = [xmin, ymin, xmax, ymax]
    return bbox

def detect_coordinate_yolo(image):
    # 设置设备
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # 加载训练好的YOLOv8模型
    yolo_model = YOLO(os.path.dirname(os.path.abspath(__file__)) + '/model/best.pt')
    yolo_model.to(DEVICE)
    # 使用YOLOv8预测ROI
    results = yolo_model(image)
    boxes = results[0].boxes.xyxy.cpu().numpy()

    # 收集文件路径及其对应的检测坐标
    if len(boxes) > 0:
        return boxes[0]
    else:
        return [-1, -1, -1, -1]

if __name__ == '__main__':
    image_path = '/home/xd/HUAWEI-CUP/mobilenetv3-master/demo/same_cat_without_face/3.png'
    bbox = yolo_test(image_path)
    print(bbox)
