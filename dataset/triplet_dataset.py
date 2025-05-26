# Copyright (c) Meta Platforms, Inc. and affiliates.

# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import os, glob, pickle, six
import random
import time

import cv2
import numpy as np
from PIL import Image
import torch
from torchvision import datasets, transforms

from timm.data.constants import \
    IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD, IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD
from timm.data import create_transform

# Image.MAX_IMAGE_PIXELS = None
section_time = {'open': 0, 'crop': 0, 'get_path': 0}
open_time = []

# crop body parts used cv2
def crop_body(img, coords):
    assert len(coords) == 4
    x0, y0, x1, y1 = [int(x) for x in coords]
    return img[y0:y1, x0:x1]


class ImagePairDataset(torch.utils.data.Dataset):
    def __init__(self, root, is_train, has_face=False, transform=None):
        """
        Args:
            root: train.txt或test.txt文件路径
            is_train: 是否是训练阶段
            has_face: 是否需要裁切面部图片
        """
        self.parent_root = root.rsplit('/', 1)[:-1][0]
        # print("parent_root: ", parent_root)
        self.class_list = []  # 保存每个类别的绝对路径
        self.imgs_list = []  # 保存每个类别中有多少个图像
        self.coordinates_index = []
        self.coordinates = []  # 面部数据坐标
        with open(self.parent_root + '/train_class.txt', 'r') as file:
            # print(root)
            # 读取数据集主文件夹
            for line in file.readlines():
                tmp = line.split('/')
                tmp[-1] = tmp[-1].rstrip('\n')
                if tmp[-1].isdigit():
                    class_folder = self.parent_root + '/' + line.rstrip('\n')
                    self.class_list.append(class_folder)
                else:
                    print("***lkx error***")
                    exit(-1)
        with open(root, 'r') as file:
            for line in file.readlines():
                self.imgs_list.append(line.replace("\n", ""))

        # 读坐标
        if has_face:
            with open(self.parent_root + '/coordinates.txt', 'r') as file:
                for line in file.readlines():
                    tmp = line.split(' ')
                    self.coordinates_index.append(tmp[0] + ' ' + tmp[1])
                    self.coordinates.append(tmp[2:])
            self.coordinates[0] = np.array(self.coordinates[0])

        if is_train:
            self.length = len(self.imgs_list)
        else:
            self.length = len(self.class_list)
        self.transform = transform
        self.is_train = is_train
        self.has_face = has_face

        # print(self.class_list)
        # print(self.imgs_list)

        # 划分测试集数据的正负样本
        if is_train is False:
            # 获取正负样本信息
            # 正样本只保存类别信息
            # 负样本保存图像信息
            len_positive = self.length // 2 + 1
            self.positive_class_path = self.class_list[:len_positive]
            negative_class_path = self.class_list[len_positive:]
            print("******" * 30)
            print(len(self.class_list))
            print("positive: ", len(self.positive_class_path), len_positive)

            self.negative_imgs_path = []
            for class_path in negative_class_path:
                class_imgs = glob.glob(os.path.join(class_path, '*'))
                self.negative_imgs_path += class_imgs

            print("negative: ", len(self.negative_imgs_path), self.length - len_positive)
            print("******" * 30)

            # 负样本图像打乱顺序
            random.shuffle(self.negative_imgs_path)

        print("共包含类别：" + str(len(self.class_list)) + "个")

    def __getitem__(self, idx):
        if self.is_train:
            return self.__get_train_data(idx)
        else:
            return self.__get_test_data(idx)

    def __len__(self):
        return self.length

    # 裁切面部数据
    def crop_face(self, img, path):
        tmp = path.split('/')
        img_index = tmp[-2] + ' ' + tmp[-1]

        coordinate_index = self.coordinates_index.index(img_index)
        coordinate = self.coordinates[coordinate_index]
        # print(self.coordinates_index)
        # print(coordinate)

        if len(coordinate) == 4:
            coordinate[-1].replace("\n", "")
            coordinate = tuple(map(float, coordinate))
        else:
            print("读取面部坐标出错！")
            exit(-1)

        none_face = (-1, -1, -1, -1)
        none_tensor = torch.zeros((3, 224, 224))
        if coordinate == none_face:
            return none_tensor, False

        face_img = img.crop(coordinate)
        return face_img, True

    def __get_train_data(self, idx):
        """
        return:
            如果裁切面部数据：返回 imgs, labels, face_imgs
            如果不裁切面部： 返回 imgs, labels
        """
        # time_1 = time.time()
        imgs_pairs = self.imgs_list[idx].split(" ")
        imgs_path = imgs_pairs[:3]
        imgs_class = imgs_pairs[-3:]
        # time_2 = time.time()
        # section_time['get_path'] += time_2-time_1

        # print("imgs_path: ", imgs_path)
        # print("imgs_class: ", imgs_class)

        imgs = []
        face_imgs = []
        for i, path in enumerate(imgs_path):
            imgs_class[i] = int(imgs_class[i])
            # time_5 = time.time()
            imgs.append(Image.open(path).convert('RGB'))
            # time_6 = time.time()
            # section_time['open'] += time_6-time_5
            # open_time.append(time_6-time_5)
            # print(path, " ++ ", time_6-time_5)
            if self.has_face:
                # 返回裁切后的面部数据
                # 如果没有面部数据，返回 None
                # time_3 = time.time()
                face_img, has_co = self.crop_face(imgs[i], path)
                # time_4 = time.time()
                # section_time['crop'] += time_4-time_3
                # print(type(face_img))
                # try:
                if self.transform is not None and has_co:
                    face_img = self.transform(face_img)
                # except RuntimeError:
                #     print("-----------")
                face_imgs.append(face_img)
            if self.transform is not None:
                imgs[i] = self.transform(imgs[i])

        if self.has_face:
            return imgs, imgs_class, face_imgs
        else:
            return imgs, imgs_class

    def __get_test_data(self, idx):
        if idx % 2 == 0:
            # assert idx <= 2752
            # print("idx: ", idx)
            idx = idx // 2
            # 读取正样本类别，随机获得两张图
            try:
                sub_imgs = glob.glob(os.path.join(self.positive_class_path[idx], '*'))
            except IndexError:
                print("***idx: ", idx)
                exit()

            imgs_path = random.sample(sub_imgs, 2)
            img_1 = Image.open(imgs_path[0]).convert('RGB')
            img_2 = Image.open(imgs_path[1]).convert('RGB')
            is_same = torch.tensor(1)

            if self.has_face:
                face_1, has_co_1 = self.crop_face(img_1, imgs_path[0])
                face_2, has_co_2 = self.crop_face(img_2, imgs_path[1])

        else:
            # 负样本按顺序读
            img_1 = Image.open(self.negative_imgs_path[idx - 1]).convert('RGB')
            img_2 = Image.open(self.negative_imgs_path[idx]).convert('RGB')

            # 但也有可能读出来的两个类别一样，因此额外判断一下
            class_1 = self.negative_imgs_path[idx - 1].split('/')[-2]
            class_2 = self.negative_imgs_path[idx].split('/')[-2]
            if class_1 == class_2:
                is_same = torch.tensor(1)
            else:
                is_same = torch.tensor(0)

            if self.has_face:
                face_1, has_co_1 = self.crop_face(img_1, self.negative_imgs_path[idx - 1])
                face_2, has_co_2 = self.crop_face(img_2, self.negative_imgs_path[idx])

        if self.transform is not None:
            img_1 = self.transform(img_1)
            img_2 = self.transform(img_2)
            try:
                if self.has_face:
                    if has_co_1:
                        face_1 = self.transform(face_1)
                    if has_co_2:
                        face_2 = self.transform(face_2)
            except TypeError:
                print(face_1)
                print(face_2)
                exit(-1)

        if self.has_face:
            return [img_1, img_2], [face_1, face_2], is_same
        else:
            return [img_1, img_2], is_same


class AlignImagePairDataset(torch.utils.data.Dataset):
    def __init__(self, root, is_train, seed=0, has_face=False, has_body=True, transform=None):
        """
        Args:
            root: train.txt或test.txt文件路径
            is_train: 是否是训练阶段
            has_face: 是否需要裁切面部图片
        """
        self.parent_root = root.rsplit('/', 1)[:-1][0]
        # print("parent_root: ", parent_root)
        self.class_list = []  # 保存每个类别的绝对路径
        self.imgs_list = []  # 保存每个类别中有多少个图像
        # 面部与身体数据坐标
        self.coordinates_index = []
        self.face_coords = []
        self.body_coords = []
        # 读取关键点数据
        self.key_points_index = []
        self.key_points = []

        if is_train:
            data_path = self.parent_root.rsplit('/')[0] + '/classes_train.txt'
            if not os.path.exists(data_path):
                data_path = self.parent_root + '/train_class.txt'
        else:
            data_path = self.parent_root.rsplit('/')[0] + '/classes_test.txt'
            if not os.path.exists(data_path):
                data_path = root

        # 获取所有类别标签
        with open(data_path, 'r') as file:
            # 读取数据集主文件夹
            for line in file.readlines():
                tmp = line.rstrip('\n')
                if not tmp.endswith('.txt'):
                    # 获取类别标签
                    class_folder = self.parent_root + '/' + line.rstrip('\n')
                    # 对类别去重，加入List中
                    if class_folder not in self.class_list:
                        self.class_list.append(class_folder)
                else:
                    print("***lkx error***")
                    exit(-1)
        print(self.class_list)

        # 如果是训练，获取所有图片对
        # 如果是测试，获取所有图片路径
        if is_train:
            with open(root, 'r') as file:
                for line in file.readlines():
                    self.imgs_list.append(line.replace("\n", ""))
        else:
            for path in self.class_list:
                img_path = glob.glob(os.path.join(path, '*'))
                self.imgs_list.extend(img_path)

        # 读坐标
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

        # 如果是训练，长度是图片对的个数
        # 如果是测试，长度是图片样本的个数
        self.length = len(self.imgs_list)
        self.transform = transform
        self.is_train = is_train
        self.has_face = has_face
        self.has_body = has_body
        self.seed = seed

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

        # 进行脸部对齐
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

        # src = marks[[0, 1, 2, 5]]

        key_marks = np.array([int(width / 4), int(height / 5 * 3),
                              int(width / 4 * 3), int(height / 5 * 3),
                              int(width / 2), height - 5,
                              int(width / 4), int(height / 4),
                              int(width / 4 * 3), int(height / 4)
                              ]).reshape([-1, 2])

        # dst = key_marks[[0, 1, 2, 5]]

        # 获取变换矩阵
        M,_ = cv2.findHomography(marks, key_marks)
        # 执行变换操作
        transformed = cv2.warpPerspective(image, M, (width, height), borderValue=0.0)

        if flag:
            image = torch.zeros((3, 224, 224))

        return image, transformed

    def __get_train_data(self, idx):
        """
        return:
            如果裁切面部数据：返回 imgs, labels, face_imgs
            如果不裁切面部： 返回 imgs, labels
        """
        imgs_pairs = self.imgs_list[idx].split(" ")
        imgs_path = imgs_pairs[:3]
        imgs_class = imgs_pairs[-3:]

        body_imgs = []
        face_imgs = []
        for i, path in enumerate(imgs_path):
            # Step1: 获取图像的：1.类别标签；2.类别+图片名，用于作为索引值查找坐标
            imgs_class[i] = int(imgs_class[i])
            tmp = path.split('/')
            img_index = tmp[-2] + ' ' + tmp[-1]

            # Step2: 读取图片
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
                # 保存脸部数据
                face_imgs.append(align_face)
            else:
                # 如果不要脸部数据，直接读图像，并裁切身体部分
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
            return body_imgs, imgs_class, face_imgs
        else:
            return body_imgs, imgs_class

    def __get_test_data(self, idx):
        # 设置种子，控制负样本选取
        if self.seed > 0:
            random.seed(self.seed)

        # 样本对中的第一张图：按顺序读出
        imgs_path = [self.imgs_list[idx]]
        class_1 = imgs_path[0].split('/')[-2]
        flag = 1

        # 读样本对的第二张图——偶数读相同类别的，奇数读不同类别的
        if idx % 2 == 0:
            # 读取前后图像
            if idx < self.length - 1 and self.imgs_list[idx+1].split('/')[-2] == class_1:
                imgs_path.append(self.imgs_list[idx+1])
            else:
                imgs_path.append(self.imgs_list[idx-1])

            flag = 1
        else:
            delta_index = random.randint(6, self.length // 3)
            if idx <= self.length / 2:
                negative_idx = idx + delta_index
            else:
                negative_idx = idx - delta_index
            imgs_path.append(self.imgs_list[negative_idx])
            flag = 0

        # 但也有可能读出来的两个类别一样，因此额外判断一下
        class_1 = imgs_path[0].split('/')[-2]
        class_2 = imgs_path[1].split('/')[-2]
        if class_1 == class_2:
            is_same = torch.tensor(1)
        else:
            is_same = torch.tensor(0)

        # if flag != is_same:
        #     print(f'ERROR: idx={idx} | {imgs_path}')

        # print(f'positive: {self.positive_num} | negative: {self.negative_num}')
        # print(imgs_path)
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


class MultiPoseDogDataset(torch.utils.data.Dataset):
    def __init__(self, root, is_train, has_face=False, transform=None):
        """
        Args:
            root: train.txt或test.txt文件路径
            is_train: 是否是训练阶段
            has_face: 是否需要裁切面部图片
        """
        self.parent_root = root.rsplit('/', 1)[:-1][0]
        # print("parent_root: ", parent_root)
        self.class_list = []  # 保存类别名
        self.coordinates_index = []
        self.coordinates = []  # 面部数据坐标
        self.imgs_list = glob.glob(os.path.join(root, '*'))  # 保存所有图像
        self.imgs_list = sorted(self.imgs_list, key=lambda x: self._get_class_name(x))

        for img_path in self.imgs_list:
            class_name = self._get_class_name(img_path)
            if class_name not in self.class_list:
                self.class_list.append(class_name)
        if is_train:
            dataset_name = 'train'
        else:
            dataset_name = 'gallery'

        if has_face:
            with open(self.parent_root + '/coordinates.txt', 'r') as file:
                for line in file.readlines():
                    tmp = line.split(' ')
                    if tmp[0] == dataset_name:
                        self.coordinates_index.append(tmp[1])
                        self.coordinates.append(tmp[2:])
            self.coordinates[0] = np.array(self.coordinates[0])

        self.length = len(self.imgs_list)

        self.transform = transform
        self.is_train = is_train
        self.has_face = has_face

        print("共包含类别：" + str(len(self.class_list)) + "个")

    def __getitem__(self, idx):
        if self.is_train:
            return self.__get_train_data(idx)
        else:
            return self.__get_test_data(idx)

    def __len__(self):
        return self.length

    # 裁切面部数据
    def crop_face(self, img, path):
        tmp = path.split('/')
        img_index = tmp[-1]

        coordinate_index = self.coordinates_index.index(img_index)
        coordinate = self.coordinates[coordinate_index]
        # print(self.coordinates_index)
        # print(coordinate)

        if len(coordinate) == 4:
            coordinate[-1].replace("\n", "")
            coordinate = tuple(map(float, coordinate))
        else:
            print("读取面部坐标出错！")
            exit(-1)

        none_face = (-1, -1, -1, -1)
        none_tensor = torch.zeros((3, 224, 224))
        if coordinate == none_face:
            return none_tensor, False

        face_img = img.crop(coordinate)
        return face_img, True

    def get_class_num(self):
        return len(self.class_list)

    def _get_class_name(self, path):
        return int(path.split('/')[-1].split('_')[0])

    def __get_train_data(self, idx):
        """
        return:
            如果裁切面部数据：返回 imgs, labels, face_imgs
            如果不裁切面部： 返回 imgs, labels
        """
        anchor_path = self.imgs_list[idx]
        anchor_class = self._get_class_name(anchor_path)
        if idx == len(self.imgs_list) - 1 or self._get_class_name(self.imgs_list[idx + 1]) != anchor_class:
            positive_path = self.imgs_list[idx - 1]
        else:
            positive_path = self.imgs_list[idx + 1]

        # 读取两个图像，第一个图作为anchor，第二个图作为正样本
        img_1 = Image.open(anchor_path).convert('RGB')
        img_2 = Image.open(positive_path).convert('RGB')

        # 读取负样本，保证与正样本类别不同
        delta_index = random.randint(6, self.length // 3)
        if idx <= self.length / 2:
            negative_idx = idx + delta_index
        else:
            negative_idx = idx - delta_index
        negative_path = self.imgs_list[negative_idx]
        imgs_path = [anchor_path, positive_path, negative_path]

        img_3 = Image.open(negative_path).convert('RGB')

        imgs = [img_1, img_2, img_3]
        labels = []
        face_imgs = []

        for i, path in enumerate(imgs_path):
            label = self._get_class_name(path)
            class_idx = self.class_list.index(label)
            labels.append(class_idx)

            if self.has_face:
                # 返回裁切后的面部数据
                # 如果没有面部数据，返回 None
                face_img, has_co = self.crop_face(imgs[i], path)
                if self.transform is not None and has_co:
                    face_img = self.transform(face_img)
                face_imgs.append(face_img)
            if self.transform is not None:
                imgs[i] = self.transform(imgs[i])

        if self.has_face:
            return imgs, labels, face_imgs
        else:
            return imgs, labels

    def __get_test_data(self, idx):
        if idx % 2 == 0:
            if idx < self.length - 1:
                imgs_path = self.imgs_list[idx:idx + 2]
            else:
                imgs_path = self.imgs_list[idx - 2:idx]
            img_1 = Image.open(imgs_path[0]).convert('RGB')
            img_2 = Image.open(imgs_path[1]).convert('RGB')
        else:
            img_path_1 = self.imgs_list[idx]
            if idx <= self.length:
                img_path_2 = random.choice(self.imgs_list[self.length // 2 + 1:])
            else:
                img_path_2 = random.choice(self.imgs_list[:self.length // 2])

            img_1 = Image.open(img_path_1).convert('RGB')
            img_2 = Image.open(img_path_2).convert('RGB')
            imgs_path = [img_path_1, img_path_2]

        # 但也有可能读出来的两个类别一样，因此额外判断一下
        class_1 = self._get_class_name(imgs_path[0])
        class_2 = self._get_class_name(imgs_path[1])
        if class_1 == class_2:
            is_same = torch.tensor(1)
        else:
            is_same = torch.tensor(0)

        if self.has_face:
            face_1, has_co_1 = self.crop_face(img_1, imgs_path[0])
            face_2, has_co_2 = self.crop_face(img_2, imgs_path[1])

        if self.transform is not None:
            img_1 = self.transform(img_1)
            img_2 = self.transform(img_2)
            try:
                if self.has_face:
                    if has_co_1:
                        face_1 = self.transform(face_1)
                    if has_co_2:
                        face_2 = self.transform(face_2)
            except TypeError:
                print(face_1)
                print(face_2)
                exit(-1)

        if self.has_face:
            return [img_1, img_2], [face_1, face_2], is_same
        else:
            return [img_1, img_2], is_same


def build_dataset(args, is_train, has_face=False, has_body=True):
    transform = build_transform(is_train, args)

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

    if args.data_set == 'CIFAR':
        dataset = datasets.CIFAR100(args.data_path, train=is_train, transform=transform, download=True)
        nb_classes = 100
    elif args.data_set == 'IMNET':
        print("reading from datapath", args.data_path)
        root = os.path.join(args.data_path, 'train' if is_train else 'val')
        dataset = datasets.ImageFolder(root, transform=transform)
        nb_classes = 1000
    elif args.data_set == 'PETFINDER':
        print("reading from datapath", args.data_path)
        if is_train:
            path = args.data_path + '/train.txt'
        else:
            path = args.data_path + '/test.txt'
        dataset = AlignImagePairDataset(path, is_train, args.seed, has_face, has_body, transform=transform)
        nb_classes = 1000
    elif args.data_set == 'PFALL':
        print("reading from datapath", args.data_path)
        if not os.path.exists(args.data_path):
            print("*** Dataset path is not exited!")
        with open(args.data_path, 'r') as file:
            path = [x.replace('\n', '').rstrip() for x in file.readlines()]

        dataset = AllAlignImagePairDataset(path, is_train, has_face, has_body, transform=transform)
        nb_classes = 1000
    elif args.data_set == 'MPDD':
        print("reading from datapath", args.data_path)
        if is_train:
            path = args.data_path + '/train'
        else:
            path = args.data_path + '/gallery'
        dataset = MultiPoseDogDataset(path, is_train, has_face, transform=transform)
        nb_classes = dataset.get_class_num()
    elif args.data_set == "image_folder":
        root = args.data_path if is_train else args.eval_data_path
        dataset = datasets.ImageFolder(root, transform=transform)
        nb_classes = args.nb_classes
        assert len(dataset.class_to_idx) == nb_classes
    else:
        raise NotImplementedError()
    print("Number of the class = %d" % nb_classes)

    return dataset, nb_classes


def build_transform(is_train, args):
    resize_im = args.input_size > 32
    imagenet_default_mean_and_std = args.imagenet_default_mean_and_std
    mean = IMAGENET_INCEPTION_MEAN if not imagenet_default_mean_and_std else IMAGENET_DEFAULT_MEAN
    std = IMAGENET_INCEPTION_STD if not imagenet_default_mean_and_std else IMAGENET_DEFAULT_STD

    if is_train:
        # this should always dispatch to transforms_imagenet_train
        transform = create_transform(
            input_size=args.input_size,
            is_training=True,
            color_jitter=args.color_jitter,
            auto_augment=args.aa,
            interpolation=args.train_interpolation,
            re_prob=args.reprob,
            re_mode=args.remode,
            re_count=args.recount,
            mean=mean,
            std=std,
        )
        if not resize_im:
            transform.transforms[0] = transforms.RandomCrop(
                args.input_size, padding=4)
        return transform

    t = []
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

        train = AlignImagePairDataset('/home/xd/HUAWEI-CUP/petfinder_extra_cats(processed)/train.txt', True, 0, True, has_body=False)

        start = time.time()
        for i in range(0, train.length):
            imgs, labels, face_imgs = train.__getitem__(i)
            # print(imgs, face_imgs)
        end = time.time()
        all_time = end - start
        print(all_time)
    else:
        test = AlignImagePairDataset('/home/xd/HUAWEI-CUP/DogFaceNet_Dataset_224_1/after_4_bis/test.txt', False, 0, False, has_body=False)

        start = time.time()
        for i in range(0, test.length):
            img1, is_same = test.__getitem__(i)
        end = time.time()
        all_time = end - start
        print(all_time)

