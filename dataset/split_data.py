import ast
import os
import glob
import random
import re
import shutil

import numpy as np
import torch
from matplotlib import pyplot as plt


def generate_split_dataset(path, ratio=0.3, val_ratio=-1.0):
    if os.path.exists(path) is False:
        print("数据集不存在，请检查文件路径！")
        return

    # 读取所有类别信息到imgs
    imgs = glob.glob(os.path.join(path, '*'))
    imgs = sorted([img for img in imgs if not img.endswith('.txt')])
    print(imgs)

    # 获取所有类别数和训练测试类别数
    imgs_class_num = len(imgs)
    if val_ratio < 0:
        test_class_num = int(imgs_class_num * ratio)
        val_class_num = 0
    else:
        test_class_num = int(imgs_class_num * ratio)
        val_class_num = int(imgs_class_num * val_ratio)

    # 随机抽取测试类别和训练类别
    index_all = [i for i in range(imgs_class_num)]
    index_test = random.sample(range(imgs_class_num), test_class_num)
    index_test = sorted(index_test)
    index_train = [i for i in range(imgs_class_num) if i not in index_test]
    # 随机抽取验证集
    if val_class_num != 0:
        index_val = random.sample(index_train, val_class_num)
        for i in index_val:
            if i in index_train:
                index_train.remove(i)
    else:
        index_val = None
    print("test: ", index_test)
    print("test_len: ", len(index_test))
    print("****"*20)
    print("train: ", index_train)
    print("train_len: ", len(index_train))
    print("****"*20)
    print("val: ", index_val)
    if index_val is not None:
        print("val_len: ", len(index_val))
    print("****"*20)

    # 写入文件
    with open(path + '/test.txt', 'w+') as test_file:
        for i in index_test:
            test_file.write(imgs[i].split('/')[-1]+'\n')

    with open(path + '/train_class.txt', 'w+') as train_file:
        for i in index_train:
            train_file.write(imgs[i].split('/')[-1]+'\n')

    if index_val is not None:
        with open(path + '/val.txt', 'w+') as train_file:
            for i in index_val:
                train_file.write(imgs[i].split('/')[-1]+'\n')

def generate_train_pair(file_path):
    if os.path.exists(file_path) is False:
        print("数据集不存在，请检查文件路径！")
        return
    if os.path.exists(file_path+'/train_class.txt') is False:
        generate_split_dataset(image_folder)

    class_list = []
    all_imgs_path = []
    with open(file_path+'/train_class.txt', 'r') as file:
        for line in file.readlines():
            # 去重
            if line.replace("\n", "") not in class_list:
                class_list.append(line.replace("\n", ""))
                all_imgs_path.extend(glob.glob(os.path.join(file_path+'/'+class_list[-1], '*')))

    print("训练集总类别数：", len(class_list))
    len_all_imgs_path = len(all_imgs_path)

    img_pairs = []
    # 遍历所有图像地址，创建图像对
    for i, img_path in enumerate(all_imgs_path):
        # 获取 anchor 和 positive
        anchor_path = img_path
        positive_path = all_imgs_path[i+1 if i!=len_all_imgs_path-1 else i-1]
        if anchor_path.split('/')[-2] != positive_path.split('/')[-2]:
            positive_path = all_imgs_path[i-1]

        # 读取负样本，保证与正样本类别不同
        delta_index = random.randint(6, len_all_imgs_path//3)
        if i <= len_all_imgs_path/2:
            negative_idx = i + delta_index
        else:
            negative_idx = i - delta_index
        negative_path = all_imgs_path[negative_idx]

        ap_class = anchor_path.split('/')[-2]
        n_class = negative_path.split('/')[-2]

        ap_idx = class_list.index(ap_class)
        n_idx = class_list.index(n_class)

        img_pairs.append(anchor_path+" "+positive_path+" "+negative_path+" "+str(ap_idx)+" "+str(ap_idx)+" "+str(n_idx))

    with open(file_path+'/train.txt', 'w+') as f:
        for pair in img_pairs:
            f.write(pair+'\n')


def generate_feature_split_pair(path, ratio, val_ratio=-1.0):
    # 读取tensor数据
    labels = []
    features = []
    with open(path, 'r') as file:
        label = -1
        for line in file.readlines():
            tmp = line.split('+')
            label = tmp[0]
            # print(tmp[1])
            # print(tmp[1].replace('tensor(', "").replace(')', "").replace("\n", ""))
            feature = ast.literal_eval(tmp[1].replace('tensor(', "").replace(')', "").replace("\n", ""))
            # feature = torch.tensor(feature)

            if label in labels:
                idx = labels.index(label)
                features[idx].append(feature)
            else:
                labels.append(label)
                features.append([feature])

    # 划分正负样本
    # 原则：按类别对正负样本五五分，前一半类别做正样本对，后一般类别做负样本对
    len_labels = len(labels)
    labels_for_positive = labels[:len_labels//2]
    labels_for_negative = labels[len_labels//2:]
    positive_pairs = []
    negative_pairs = []

    for i in range(0, len_labels):
        if labels[i] in labels_for_positive:
            # 创建正样本对
            length = len(features[i])  # 该类别下有多少个特征
            for j, f in enumerate(features[i]):
                positive_pairs.append([1, f, features[i][(j+1)%length]])
        elif labels[i] in labels_for_negative:
            # 创建负样本对
            length = len(features[i])  # 该类别下有多少个特征
            # 遍历该类别下的所有图像
            for j, f in enumerate(features[i]):
                # 随机选一个负样本类别，不能与当前类别相同
                label_negative = random.choice(labels_for_negative)
                while labels[i] == label_negative:
                    label_negative = random.choice(labels_for_negative)

                # 获取该负样本类别在所有类别中的索引
                idx_negative = labels.index(label_negative)
                # 随机获取该类别下的一张图
                negative_feature = random.choice(features[idx_negative])
                negative_pairs.append([0, f, negative_feature])
        else:
            print(f'遗漏类别！label={labels[i]}  idx={i}  len_labels={len_labels}')
            exit(-1)

    print(labels_for_positive)
    print(labels_for_negative)
    print(positive_pairs)
    print(negative_pairs)

    # 写入train/test文件
    train_path = os.path.join(path.rsplit('/', 1)[:-1][0], 'feature_train.txt')
    test_path = os.path.join(path.rsplit('/', 1)[:-1][0], 'feature_test.txt')
    train_num = 0
    test_num = 0
    with open(train_path, 'w') as train, open(test_path, 'w') as test:
        # 对正样本划分
        len_test_positive = len(positive_pairs) * ratio
        for i, pair in enumerate(positive_pairs):
            if i <= len_test_positive:
                test.write('+'.join(map(str, pair))+'\n')
                test_num += 1
            else:
                train.write('+'.join(map(str, pair))+'\n')
                train_num += 1

        # 对负样本划分
        len_test_negative = len(negative_pairs) * ratio
        for j, pair in enumerate(negative_pairs):
            if j <= len_test_negative:
                test.write('+'.join(map(str, pair))+'\n')
                test_num += 1
            else:
                train.write('+'.join(map(str, pair))+'\n')
                train_num += 1

    print(f'数据划分完成！\n'
          f'共生成训练数据{train_num}条，保存在{train_path}中\n'
          f'测试数据{test_num}条，保存在{test_path}中\n')

def random_feature_txt():
    with open('/home/xd/HUAWEI-CUP/mobilenetv3-master/dataset/feature.txt', 'w') as file:
        for i in range(0, 200):
            x = torch.rand((1, 10))
            print(f'{i%6}+{x.tolist()}\n')
            file.write(f'{i%6}+{x.tolist()}\n')

"""
获取数据集的数据分布
"""
def visualize_data_distribution(path):
    # 获取所有子文件夹，即所有类别文件夹
    class_folder = glob.glob(os.path.join(path, '*'))

    num_in_class = []
    for folder in class_folder:
        # 跳过txt文件
        if folder.endswith('.txt'):
            continue

        sub_imgs = glob.glob(os.path.join(folder, '*'))
        len_sub_imgs = len(sub_imgs)
        num_in_class.append(len_sub_imgs)

    with open('distribution.txt', 'w') as file:
        for x in num_in_class:
            file.write(str(x)+' ')

    plt.style.use('ggplot')
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.hist(x=num_in_class, bins=10, range=(0, 8), color='steelblue')
    plt.xlabel('类别内样本个数')
    plt.ylabel('每个样本数的分布')
    plt.title('数据集分布图')
    plt.show()




if __name__ == '__main__':
    # 图像文件夹路径
    # image_folder = 'E:\竞赛\“华为杯”第六届中国研究生人工智能创新大赛\数据集\petfinder_extra_cats(processed)'
    image_folder = '/home/xd/HUAWEI-CUP/petfinder_all'

    mode = input("使用 1.[训练集/测试集] 划分\n2.[训练集/验证集/测试集] 划分\n3.生成图片对？\n4.特征集划分\n5.可视化数据集分布\n（输入1~5）:")
    mode = int(mode)
    if mode == 1:
        print("默认划分比例为7:3")
        generate_split_dataset(image_folder)
        print("数据集划分完成！")
    elif mode == 2:
        print("默认划分比例为6:2:2")
        generate_split_dataset(image_folder, ratio=0.2, val_ratio=0.2)
        print("数据集划分完成！")
    elif mode == 3:
        print("**开始生成图片对")
        generate_train_pair(image_folder)
        print("图片对生成完成！")
    elif mode == 4:
        txt_path = '/home/xd/HUAWEI-CUP/mobilenetv3-master/dataset/feature.txt'
        generate_feature_split_pair(txt_path, ratio=0.2)
    elif mode == 5:
        visualize_data_distribution(image_folder)
        # random_feature_txt()
    else:
        print("错误！请输入1或2")