# Copyright (c) Meta Platforms, Inc. and affiliates.

# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import os, glob, pickle, six
import random

import cv2
import numpy as np
from PIL import Image
import torch
from torchvision import datasets, transforms
from torch.utils import data

from timm.data.constants import \
    IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD, IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD
from timm.data import create_transform
import time
import xml.etree.ElementTree as ET


def crop_body(img, coords):
    assert len(coords) == 4
    x0, y0, x1, y1 = [int(x) for x in coords]
    return img[y0:y1, x0:x1]


"""
    Dataset read from TXT file
"""
class ClassifyFromTxtDataset(data.Dataset):

    def __init__(self, root, txt_file, roi_dir, th_image=None, th_txt=None, transforms=None, train=False, test=False, body_id=0):
        """
        主要目标： 获取所有图片的地址，并根据训练，验证，测试划分数据
        """
        self.train = train
        self.test = test
        self.transforms = transforms
        self.imgs = []
        self.labels = []
        self.img_sources = []  # 用于记录每张图片的来源目录
        self.roi_dir = roi_dir  # Directory containing ROI txt files
        self.th_txt = th_txt  # Directory containing XML files
        self.body_id = body_id

        # 加载 txt_file 文件中的数据
        with open(txt_file, 'r') as f:
            lines = f.readlines()
            for line in lines:
                if line.strip():  # 忽略空行
                    parts = line.strip().split()
                    base_filename = parts[0]
                    img_path_jpg = os.path.join(root, base_filename + '.jpg')
                    img_path_gif = os.path.join(root, base_filename + '.gif')
                    if os.path.exists(img_path_jpg):
                        self.imgs.append(img_path_jpg)
                        self.img_sources.append('root')
                    elif os.path.exists(img_path_gif):
                        self.imgs.append(img_path_gif)
                        self.img_sources.append('root')
                    else:
                        continue
                    class_id = int(parts[1]) - 1  # 假设 CLASS-ID 是第二个元素
                    self.labels.append(class_id)

        if th_image is not None and os.path.exists(th_image):
            # 在训练和测试模式下都加载 th_image 文件夹中的数据
            for class_folder in os.listdir(th_image):
                class_path = os.path.join(th_image, class_folder)
                class_id = int(class_folder.split('-')[-1]) - 1  # 假设 CLASS-ID 是 '-' 后的最后一部分
                if os.path.isdir(class_path):
                    for img_file in os.listdir(class_path):
                        if img_file.endswith(('.jpg', '.jpeg', '.png', '.gif')):  # 支持 JPEG、JPG、PNG 和 GIF 格式
                            img_path = os.path.join(class_path, img_file)
                            self.imgs.append(img_path)
                            self.img_sources.append('th_image')
                            self.labels.append(class_id)

        self.imgs = np.array(self.imgs)
        self.labels = np.array(self.labels)
        self.img_sources = np.array(self.img_sources)

    def read_roi(self, roi_path, body_id):
        with open(roi_path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                parts = line.strip().split()
                class_id = int(parts[0])
                if class_id == body_id:
                    xmin = int(parts[1])
                    ymin = int(parts[2])
                    xmax = int(parts[3])
                    ymax = int(parts[4])
                    return xmin, ymin, xmax, ymax
        return None, None, None, None  # If no ROI for the specified class

    def __getitem__(self, index):
        img_path = self.imgs[index]
        label = self.labels[index]
        img_source = self.img_sources[index]  # 获取图片的来源目录

        img = Image.open(img_path).convert('RGB')
        if img.mode == 'P':  # 检查是否是调色板模式
            img = img.convert('RGBA')  # 转换为RGBA模式
        img_width, img_height = img.size

        base_filename = os.path.basename(img_path).rsplit('.', 1)[0]
        roi_txt_path = None

        # 根据图片的来源目录选择正确的ROI文件路径
        if img_source == 'root':
            roi_txt_path = os.path.join(self.roi_dir, base_filename + '.txt')
        elif img_source == 'th_image':
            roi_txt_path = os.path.join(self.th_txt, base_filename + '.txt')

        bxmin, bymin, bxmax, bymax = None, None, None, None

        bxmin, bymin, bxmax, bymax = self.read_roi(roi_txt_path, 1)
        fxmin, fymin, fxmax, fymax = self.read_roi(roi_txt_path, 0)

        body_img = None
        face_img = None

        if bxmin is not None and bymin is not None and bxmax is not None and bymax is not None:
            bxmin = max(0, bxmin)
            bymin = max(0, bymin)
            bxmax = min(img_width, bxmax)
            bymax = min(img_height, bymax)
            body_img = img.crop((bxmin, bymin, bxmax, bymax))
        if fxmin is not None and fymin is not None and fxmax is not None and fymax is not None:
            fxmin = max(0, fxmin)
            fymin = max(0, fymin)
            fxmax = min(img_width, fxmax)
            fymax = min(img_height, fymax)
            face_img = img.crop((fxmin, fymin, fxmax, fymax))

        if self.transforms:
            if body_img is not None:
                body_img = self.transforms(body_img)
            if face_img is not None:
                face_img = self.transforms(face_img)

        return body_img, face_img, label

    def __len__(self):
        return len(self.imgs)


"""
    Dataset read from XML file
"""
class ClassifyFromXMLDataset(data.Dataset):

    def __init__(self, th_image, th_txt, transforms=None, train=False, test=False, body_id=0):
        """
        主要目标： 获取所有图片的地址，并根据训练，验证，测试划分数据
        """
        self.train = train
        self.test = test
        self.transforms = transforms
        self.imgs = []
        self.labels = []
        self.img_sources = []  # 用于记录每张图片的来源目录
        # self.roi_dir = roi_dir  # Directory containing ROI txt files
        self.th_txt = th_txt  # Directory containing XML files
        self.body_id = body_id

        # 加载 txt_file 文件中的数据
        # with open(txt_file, 'r') as f:
        #     lines = f.readlines()
        #     for line in lines:
        #         if line.strip():  # 忽略空行
        #             parts = line.strip().split()
        #             base_filename = parts[0]
        #             img_path_jpg = os.path.join(root, base_filename + '.jpg')
        #             img_path_gif = os.path.join(root, base_filename + '.gif')
        #             if os.path.exists(img_path_jpg):
        #                 self.imgs.append(img_path_jpg)
        #                 self.img_sources.append('root')
        #             elif os.path.exists(img_path_gif):
        #                 self.imgs.append(img_path_gif)
        #                 self.img_sources.append('root')
        #             else:
        #                 continue
        #             class_id = int(parts[1]) - 1  # 假设 CLASS-ID 是第二个元素
        #             self.labels.append(class_id)

        if th_image is not None and os.path.exists(th_image):
            # 在训练和测试模式下都加载 th_image 文件夹中的数据
            for class_folder in os.listdir(th_image):
                if not class_folder.endswith('txt'):
                    class_path = os.path.join(th_image, class_folder)
                    class_id = int(class_folder.split('-')[-1]) - 1  # 假设 CLASS-ID 是 '-' 后的最后一部分
                    if os.path.isdir(class_path):
                        for img_file in os.listdir(class_path):
                            if img_file.endswith(('.jpg', '.jpeg', '.png', '.gif')):  # 支持 JPEG、JPG、PNG 和 GIF 格式
                                img_path = os.path.join(class_path, img_file)
                                self.imgs.append(img_path)
                                self.img_sources.append('th_image')
                                self.labels.append(class_id)

        self.imgs = np.array(self.imgs)
        self.labels = np.array(self.labels)
        self.img_sources = np.array(self.img_sources)

    def read_roi(self, roi_path, body_id):
        with open(roi_path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                parts = line.strip().split()
                class_id = int(parts[0])
                if class_id == body_id:
                    xmin = int(parts[1])
                    ymin = int(parts[2])
                    xmax = int(parts[3])
                    ymax = int(parts[4])
                    return xmin, ymin, xmax, ymax
        return None, None, None, None  # If no ROI for the specified class

    def __getitem__(self, index):
        img_path = self.imgs[index]
        label = self.labels[index]
        img_source = self.img_sources[index]  # 获取图片的来源目录

        img = Image.open(img_path).convert('RGB')
        if img.mode == 'P':  # 检查是否是调色板模式
            img = img.convert('RGBA')  # 转换为RGBA模式
        img_width, img_height = img.size

        base_filename = os.path.basename(img_path).rsplit('.', 1)[0]
        roi_txt_path = None

        # 根据图片的来源目录选择正确的ROI文件路径
        if img_source == 'root':
            roi_txt_path = os.path.join(self.roi_dir, base_filename + '.txt')
        elif img_source == 'th_image':
            roi_txt_path = os.path.join(self.th_txt, base_filename + '.txt')

        bxmin, bymin, bxmax, bymax = None, None, None, None

        bxmin, bymin, bxmax, bymax = self.read_roi(roi_txt_path, 1)
        fxmin, fymin, fxmax, fymax = self.read_roi(roi_txt_path, 0)

        body_img = None
        face_img = None

        if bxmin is not None and bymin is not None and bxmax is not None and bymax is not None:
            bxmin = max(0, bxmin)
            bymin = max(0, bymin)
            bxmax = min(img_width, bxmax)
            bymax = min(img_height, bymax)
            body_img = img.crop((bxmin, bymin, bxmax, bymax))
        if fxmin is not None and fymin is not None and fxmax is not None and fymax is not None:
            fxmin = max(0, fxmin)
            fymin = max(0, fymin)
            fxmax = min(img_width, fxmax)
            fymax = min(img_height, fymax)
            face_img = img.crop((fxmin, fymin, fxmax, fymax))

        if self.transforms:
            if body_img is not None:
                body_img = self.transforms(body_img)
            if face_img is not None:
                face_img = self.transforms(face_img)

        return body_img, face_img, label

    def __len__(self):
        return len(self.imgs)


class IdentifyDataset(data.Dataset):
    def __init__(self, root, is_train, seed=0, has_face=False, has_body=True, transform=None):
        self.parent_root = root.rsplit('/', 1)[:-1][0]
        self.class_list = []  # 保存每个类别的绝对路径
        self.imgs_list = []  # 保存每个类别中有多少个图像
        # 面部数据坐标
        self.coordinates_index = []
        self.face_coords = []
        self.body_coords = []
        # 读取关键点数据
        self.key_points_index = []
        self.key_points = []

        # 如果是训练阶段，读取classes_train.txt或者train_class.txt，即所有训练类别
        # 如果是测试阶段，读取classes_test.txt或者root，即所有测试类别
        if is_train:
            data_path = self.parent_root.rsplit('/')[0] + '/classes_train.txt'
            if not os.path.exists(data_path):
                data_path = self.parent_root + '/train_class.txt'
        else:
            data_path = self.parent_root.rsplit('/')[0] + '/classes_test.txt'
            if not os.path.exists(data_path):
                data_path = root

        with open(data_path, 'r') as file:
            # 读取数据集主文件夹
            # index表示当前类别的索引，代替类别名作为标签使用
            index = 0
            for line in file.readlines():
                tmp = line.rstrip('\n')
                if not tmp.endswith('.txt'):
                    # 获取类别标签
                    class_folder = self.parent_root+'/'+line.rstrip('\n')
                    if class_folder not in self.class_list:
                        self.class_list.append(class_folder)
                        # 获取每个类别中的图片地址
                        sub_imgs = glob.glob(os.path.join(class_folder, '*'))
                        for x in sub_imgs:
                            self.imgs_list.append((x, index))
                        index += 1
                else:
                    print("***lkx error***")
                    exit(-1)

        self.length = len(self.imgs_list)
        self.transform = transform
        self.is_train = is_train
        self.has_face = has_face
        self.has_body = has_body
        self.seed = seed

        # 读坐标和关键点
        if has_face:
            with open(self.parent_root + '/coordinates.txt', 'r') as file:
                for line in file.readlines():
                    tmp = line.split(' ')
                    self.coordinates_index.append(tmp[0] + ' ' + tmp[1])
                    self.face_coords.append(tmp[2:6])
                    self.body_coords.append(tmp[6:])
            with open(self.parent_root + '/landmark.txt', 'r') as file:
                for line in file.readlines():
                    tmp = [x.replace('[', '').replace(']', '').replace('\n', '').replace(')', '') for x in line.split(' ')]
                    self.key_points_index.append(tmp[0] + ' ' + tmp[1])
                    self.key_points.append(tmp[2:])

        print("共包含类别：" + str(len(self.class_list)) + "个")

    def __getitem__(self, idx):
        if self.is_train:
            return self.__get_train_data(idx)
        else:
            return self.__get_test_data(idx)

    def __len__(self):
        return self.length

    # 裁切面部数据
    def all_align(self, img_index, path):
        # 读取关键点
        key_point_index = self.key_points_index.index(img_index)
        key_point = [float(x.rstrip(',')) for x in self.key_points[key_point_index]]

        if self.has_body:
            """如果有body，则进行面部裁剪和对齐"""
            # 读取面部和身体坐标
            coordinate_index = self.coordinates_index.index(img_index)
            face_coord = self.face_coords[coordinate_index]
            body_coord = self.body_coords[coordinate_index]
            # 转换坐标格式为list
            if len(face_coord) == 4 and len(body_coord) == 4:
                face_coord[-1].replace("\n", "")
                face_coord = list(map(float, face_coord))
                body_coord[-1].replace("\n", "")
                body_coord = list(map(float, body_coord))
            else:
                print("读取面部/身体坐标出错！")
                exit(-1)

            # 如果脸部为空
            none_face = [-1, -1, -1, -1]
            none_tensor = torch.zeros((3, 224, 224))
            if face_coord == none_face:
                img = cv2.imread(path)
                # 裁切身体部分
                img = crop_body(img, body_coord)
                return img, none_tensor, False
        else:
            """如果没有body，只需要对齐即可，因此面部坐标就是图像本身尺寸大小，此时传递特殊的coordinate"""
            face_coord = [-1, -1, -1, -1]

        img, align_img = self.one_align(path, face_coord, key_point)
        # 裁切身体部分
        img = crop_body(img, body_coord)
        return img, align_img, True

    def one_align(self, image_path, bbox, key_points):
        image = cv2.imread(image_path)
        bbox = np.array(bbox)
        # 特殊判断，判断是否有身体数据
        none_body = np.array([-1, -1, -1, -1])
        flag = 0

        if (bbox == none_body).all():
            # 如果没有身体数据，坐标设置为图像长宽
            height, width, _ = image.shape
            # 设置标记信息
            flag = 1
        else:
            width = int(bbox[2] - bbox[0])
            height = int(bbox[3] - bbox[1])

        marks = np.array(key_points).reshape([-1, 2])

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

        if flag:
            image = torch.zeros((3, 224, 224))

        return image, transformed

    def __get_train_data(self, idx):
        # 获取idx类别的数据
        path, label_index = self.imgs_list[idx]
        tmp = path.split('/')
        img_index = tmp[-2] + ' ' + tmp[-1]

        # 读取图片
        if self.has_face:
            # 如果需要脸部数据，则调用self.all_align函数
            body_img, align_face, has_co = self.all_align(img_index, path)
            # 只有有身体数据时才转换
            if self.has_body:
                body_img = Image.fromarray(cv2.cvtColor(body_img, cv2.COLOR_BGR2RGB))
            # 对脸部数据做transform
            if has_co:
                align_face = Image.fromarray(cv2.cvtColor(align_face, cv2.COLOR_BGR2RGB))
                if self.transform is not None:
                    align_face = self.transform(align_face)
        else:
            # 如果不要脸部数据，直接读图像
            body_img = Image.open(path).convert('RGB')
            # 获取身体坐标
            coordinate_index = self.coordinates_index.index(img_index)
            body_coord = self.body_coords[coordinate_index]
            # 裁切身体数据
            body_img = crop_body(body_img, body_coord)

        if self.transform is not None and self.has_body:
            body_img = self.transform(body_img)

        if self.has_face:
            return body_img, label_index, align_face
        else:
            return body_img, label_index

    def __get_test_data(self, idx):
        # 设置种子，控制负样本选取
        if self.seed > 0:
            random.seed(self.seed)

        # 样本对中的第一张图：按顺序读出
        (imgs_path, class_1) = self.imgs_list[idx]

        imgs_path = [imgs_path]
        if idx % 2 == 0:
            # 读取前后图像
            if idx < self.length - 1 and self.imgs_list[idx+1][1] == class_1:
                imgs_path.append(self.imgs_list[idx+1][0])
            else:
                imgs_path.append(self.imgs_list[idx-1][0])
            flag = 1
        else:
            delta_index = random.randint(6, self.length // 3)
            if idx <= self.length / 2:
                negative_idx = idx + delta_index
            else:
                negative_idx = idx - delta_index
            imgs_path.append(self.imgs_list[negative_idx][0])
            flag = 0

        # 但也有可能读出来的两个类别一样，因此额外判断一下
        class_1 = imgs_path[0].split('/')[-2]
        class_2 = imgs_path[1].split('/')[-2]
        if class_1 == class_2:
            is_same = torch.tensor(1)
        else:
            is_same = torch.tensor(0)

        if flag != is_same:
            print(f'ERROR: idx={idx} | {imgs_path}')

        face_imgs = []
        body_imgs = []
        for path in imgs_path:
            if self.has_face:
                tmp = path.split('/')
                img_index = tmp[-2] + ' ' + tmp[-1]
                # 如果需要脸部数据，则调用self.all_align函数
                body_img, align_face, has_co = self.all_align(img_index, path)
                if self.has_body:
                    body_img = Image.fromarray(cv2.cvtColor(body_img, cv2.COLOR_BGR2RGB))
                # 对脸部数据做transform
                if has_co:
                    align_face = Image.fromarray(cv2.cvtColor(align_face, cv2.COLOR_BGR2RGB))
                    if self.transform is not None:
                        align_face = self.transform(align_face)
                # 保存脸部数据
                face_imgs.append(align_face)
            else:
                # 如果不要脸部数据，直接读图像
                body_img = Image.open(path).convert('RGB')
                # 获取身体坐标
                coordinate_index = self.coordinates_index.index(img_index)
                body_coord = self.body_coords[coordinate_index]
                # 裁切身体数据
                body_img = crop_body(body_img, body_coord)

            if self.transform is not None and self.has_body:
                body_img = self.transform(body_img)
            # Step3: 保存原始图像
            body_imgs.append(body_img)

        if self.has_face:
            return body_imgs, face_imgs, is_same
        else:
            return body_imgs, is_same



def build_dataset(args, is_train, has_face=False, has_body=True):
    """
    Args:
        args: 包含各种参数信息。以下列出本函数中使用的字段、类型和取值。
            args.data_set[str]: 标识使用什么数据集。可取：['OXFORD_PET', 'DOGFACENET', 'THU_DOGS', 'STANFORD_DOGS', 'PETFINDER']
            args.data_path[str]: 标识数据集的路径。
            args.nb_classes[int]: 类别数。
            args.seed[int]: 种子标志。
    """
    # 品种分类和个体识别的transform方法不同
    if args.data_set in ['PETFINDER', 'DOGFACENET']:
        # 如果是个体识别的数据集
        transform = build_transform(False, args)
    else:
        # 如果是品种分类的数据集
        transform = build_transform(True, args)

    print("Transform = ")
    if isinstance(transform, tuple):
        for trans in transform:
            print(" - - - - - - - - - - ")
            for t in trans.transforms:
                print(t)
    else:
        for t in transform.transforms:
            print(t)
    print("---------------------------")

    if args.data_set == 'OXFORD_PET':
        dataset = ClassifyFromTxtDataset(
            args.data_path, args.data_path + '/test.txt', args.data_path+'/Oxford Pet-txt',
            transforms=transform, test=True, body_id=0)
        nb_classes = 100
    elif args.data_set == 'PETFINDER' or args.data_set == 'DOGFACENET':
        print("reading from datapath", args.data_path)
        if is_train:
            path = args.data_path + '/train.txt'
        else:
            path = args.data_path + '/test.txt'
        dataset = IdentifyDataset(path, is_train, args.seed, has_face, has_body, transform=transform)
        nb_classes = 1000
    elif args.data_set == "THU_DOGS":
        dataset = ClassifyFromXMLDataset(
            args.data_path, args.data_path+'/test-Tsinghua Dogs-txt',
            transforms=transform, test=True, body_id=0)
        nb_classes = 100
    elif args.data_set == "STANFORD_DOGS":
        dataset = ClassifyFromXMLDataset(
            args.data_path, args.data_path+'/test-Stanford Dogs-txt',
            transforms=transform, test=True, body_id=0)
        nb_classes = 100
    else:
        raise NotImplementedError()
    # print("Number of the class = %d" % nb_classes)

    return dataset, nb_classes


def build_transform(is_classify, args):
    resize_im = args.input_size > 32
    imagenet_default_mean_and_std = args.imagenet_default_mean_and_std
    mean = IMAGENET_INCEPTION_MEAN if not imagenet_default_mean_and_std else IMAGENET_DEFAULT_MEAN
    std = IMAGENET_INCEPTION_STD if not imagenet_default_mean_and_std else IMAGENET_DEFAULT_STD
    t = []

    if is_classify:
        t = [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ]
    else:
        if resize_im:
            # warping (no cropping) when evaluated at 384 or larger
            if args.input_size >= 384:
                t.append(
                transforms.Resize((args.input_size, args.input_size),
                                interpolation=transforms.InterpolationMode.BICUBIC),
            )
                print(f"Warping {args.input_size} size input images...")
            else:
                if args.crop_pct is None:
                    args.crop_pct = 224 / 256
                size = int(args.input_size / args.crop_pct)
                t.append(
                    # to maintain same ratio w.r.t. 224 images
                    transforms.Resize(size, interpolation=transforms.InterpolationMode.BICUBIC),
                )
                t.append(transforms.CenterCrop(args.input_size))

        t.append(transforms.ToTensor())
        t.append(transforms.Normalize(mean, std))
    return transforms.Compose(t)


if __name__ == '__main__':

    is_train = int(input('进行测试还是训练？输入0或1：'))
    if is_train:

        train = IdentifyDataset('/home/xd/HUAWEI-CUP/petfinder_extra_cats(processed)/train.txt', True, 0, True, has_body=False)

        start = time.time()
        for i in range(0, train.length):
            imgs, labels, face_imgs = train.__getitem__(i)
            # print(imgs, face_imgs)
        end = time.time()
        all_time = end - start
        print(all_time)
    else:
        test = IdentifyDataset('/home/xd/HUAWEI-CUP/petfinder_extra_cats(processed)/test.txt', False, 0, False, has_body=False)

        start = time.time()
        for i in range(0, test.length):
            img1, is_same = test.__getitem__(i)
        end = time.time()
        all_time = end - start
        print(all_time)