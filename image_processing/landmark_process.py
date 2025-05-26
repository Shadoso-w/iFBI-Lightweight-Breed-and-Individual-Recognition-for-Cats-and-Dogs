import os
import torch

from landmark_test import landmark_test

# path = '/home/wenli/Linux_Gutai2/HUAWEI-CUP/DogFaceNet_Dataset_224_1/after_4_bis'
path = '/home/xd/HUAWEI-CUP/DogFaceNet_Dataset_Large/images_2'
output_path = 'cats'  # 保存坐标的文件路径
os.makedirs(output_path, exist_ok=True)
coords_file_path = os.path.join(output_path, 'landmark_5_dogface_large.txt')

# image_ext = ['jpg', 'jpeg', 'png', 'bmp', 'gif']

# 收集所有文件路径及其对应的检测坐标
coords = []
count = 0
# 递归遍历子文件夹中的图片
for root, dirs, files in os.walk(path):
    for file in files:
        if file.endswith('.jpg'):
            img_path = os.path.join(root, file)
                
            results = landmark_test(img_path)
            # indexes_to_remove = {6, 7, 12, 13}
            # results_5 = [item for idx, item in enumerate(results) if idx not in indexes_to_remove]
            # 获取大文件名和小文件名
            parent_dir = os.path.basename(root)

            # 收集文件路径及其对应的检测坐标
            coords.append((parent_dir, file, results))
            # print(coords)
            count += 1
            print(count)

# for root, dirs, files in os.walk(path):
#     for file in files:
#         if file.endswith('.gif'):
#             img_path = os.path.join(root, file)
            
#             # 调用landmark函数获取检测结果
#             results = landmark(img_path)
#             indexes_to_remove = {6, 7, 12, 13}
#             results_5 = [item for idx, item in enumerate(results) if idx not in indexes_to_remove]   
            
#             # 收集文件路径及其对应的检测坐标
#             coords.append((file, results_5))

#             count += 1
#             print(count)

# 按路径排序
coords.sort()

# 写入坐标信息到文件
with open(coords_file_path, 'w') as coords_file:
    for coord in coords:
        parent_dir, file, results = coord
        coords_file.write(f"{parent_dir} {file} {results}\n")

print(f'检测坐标已保存到: {coords_file_path}')