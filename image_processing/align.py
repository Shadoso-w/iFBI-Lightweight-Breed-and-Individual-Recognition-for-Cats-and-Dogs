import numpy as np
import cv2
# 透视变换 + 对应的点（两个耳朵尖+嘴巴+左眼）

from .landmark_test import landmark
from .yolo_test import yolo_test

def align(image_path, bbox, key_points):

    image = cv2.imread(image_path)
    image1 = image
    image2 = image
    # cv2.imshow("image", image)
    bbox = np.array(bbox)
    # cv2.rectangle(image1, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 0, 255), thickness = 1)

    width = int(bbox[2] - bbox[0])
    height = int(bbox[3] - bbox[1])

    marks = np.array(key_points).reshape([-1, 2])

    # for i in range(marks.shape[0]):
    #     x, y = marks[i]
    #     image_with_box_keypoints = cv2.circle(image2, (int(x), int(y)), 3, (0, 255, 0), thickness=-1)
    # cv2.imwrite('99_keypoints.py', image)

    key_marks = np.array([int(width / 4), int(height / 5 * 3), 
                        int(width / 4 * 3), int(height / 5 * 3),
                        int(width / 2), height - 5, 
                        int(width / 4), int(height / 4), 
                        int(width / 4 * 3), int(height / 4)
                        ]).reshape([-1, 2])
    

    # 获取变换矩阵
    M,_ = cv2.findHomography(marks, key_marks)

    # 执行变换操作
    transformed = cv2.warpPerspective(image, M, (width, height), borderValue=0.0)
    # for i in range(key_marks.shape[0]):
    #     x, y = key_marks[i]
    #     cv2.circle(transformed, (int(x), int(y)), 3, (0, 0, 255), thickness=-1)
    
    # return image_with_box_keypoints, transformed
    return transformed

if __name__ == '__main__':

    image_path = '99.png'
    bbox = yolo_test(image_path)
    print(bbox) 
    key_points = landmark(image_path)
    indexes_to_remove = {6, 7, 12, 13}
    key_points_5 = [item for idx, item in enumerate(key_points) if idx not in indexes_to_remove]
    print(key_points_5) 
    # image_with_box_keypoints, transformed = align(image_path, bbox, key_points_5)
    transformed = align(image_path, bbox, key_points_5)

    # cv2.imwrite('99_keypoints.png', image_with_box_keypoints)
    output_path = '99_align.png'
    cv2.imwrite(output_path, transformed)
