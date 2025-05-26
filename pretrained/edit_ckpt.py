import torch
import argparse
import time
import utils

from model.mobilenetv3 import MobileNetV3_Small, MobileNetV3_Large
from model.facenet import Facenet
from model.fusion_mobilenet_temp import Fusion_MobileNetV3, Fusion_MobileNetV3_inference
from model.classifier import Identity_Classifier, Cosine_Classifier
import torch.nn as nn

# from dataset.single_dataset import build_dataset # for feature extract
from dataset.triplet_dataset import build_dataset
from engine import evaluate, extract_feature

import numpy as np

import torch.nn.functional as F
from torch.nn import init

from torchvision.models.mobilenetv3 import mobilenet_v3_small, mobilenet_v3_large


from pathlib import Path
from collections import OrderedDict

import sys
sys.path.insert(0, './model')

class CrossAttention(nn.Module):
    def __init__(self, mode=None):
        super().__init__()
        assert mode is not None
        if mode == 'Self Attention':
            from fightingcv_attention.attention.SelfAttention import ScaledDotProductAttention
            self.attention = ScaledDotProductAttention(d_model=960, d_k=960, d_v=960,  h=8)
        elif mode == 'Simplified Self Attention':
            from fightingcv_attention.attention.SimplifiedSelfAttention import SimplifiedScaledDotProductAttention
            self.attention = SimplifiedScaledDotProductAttention(d_model=960, h=8)
        elif mode == 'Efficient Multi-Head Self-Attention':
            from fightingcv_attention.attention.EMSA import EMSA
            self.attention = EMSA(d_model=960, d_k=960, d_v=960,  h=8, H=8, W=8,ratio=2,apply_transform=True)
        elif mode == 'MUSE Attention':
            from fightingcv_attention.attention.MUSEAttention import MUSEAttention
            self.attention = MUSEAttention(d_model=960, d_k=960, d_v=960,  h=8)
        elif mode == 'UFO Attention':
            from fightingcv_attention.attention.UFOAttention import UFOAttention
            self.attention = UFOAttention(d_model=960, d_k=960, d_v=960,  h=8)
        else:
            print("Please choose supported attention")

class Identity_Layer(nn.Module):
    def __init__(self, num_classes=1000, dropout_keep_prob=0.5, embedding_size=1024, alpha=0.5):
        super(Identity_Layer, self).__init__()

        # fusion module
        # self.fusion = CrossAttention(mode='Simplified Self Attention')

        # identity layer
        self.avg = nn.AdaptiveAvgPool2d((1, 1))
        self.Dropout = nn.Dropout(1 - dropout_keep_prob)
        self.fc1 = nn.Linear(960, embedding_size)
        self.fc2 = nn.Linear(embedding_size, embedding_size)
        self.last_bn = nn.BatchNorm1d(embedding_size, eps=0.001, momentum=0.1, affine=True)

        self.init_params()

    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

def del_other(model, keep_layer_name):
    new_name = []
    for (name, param) in model.named_parameters():
        if name in keep_layer_name:
            pass
        else:
            new_name.append(name)

    return new_name

def main():
    model = Fusion_MobileNetV3()

    face_checkpoint = torch.load(face_finetune, map_location='cpu')
    # face_checkpoint = face_checkpoint.state_dict()
    print("Load face ckpt from %s" % face_finetune)

    body_checkpoint = torch.load(body_finetune, map_location='cpu')
    # body_checkpoint = body_checkpoint.state_dict()
    print("Load body ckpt from %s" % body_finetune)

    model_checkpoint = torch.load(total_ckpt, map_location='cpu')
    model_checkpoint = model_checkpoint['model']
    print("Load model ckpt from %s" % total_ckpt)

    # checkpoint = OrderedDict(list(face_checkpoint.items()) + list(body_checkpoint.items()))
    #
    # utils.load_state_dict(model, model_checkpoint, prefix=None)

    new_model = Identity_Layer(num_classes=7666)

    # keep_layer_name = ['fc1.weight', 'fc1.bias', 'fc2.weight', 'last_bn.running_mean', 'last_bn.running_var', 'fc2.bias', 'last_bn.weight', 'last_bn.bias', 'classifier.weight', 'classifier.bias']
    keep_layer_name = ['fc1.weight', 'fc1.bias', 'fc2.weight', 'fc2.bias', 'last_bn.weight', 'last_bn.bias', 'last_bn.running_mean', 'last_bn.running_var', 'last_bn.num_batches_tracked']
    # keep_layer_name = ['fusion.attention.fc_o.weight', 'fusion.attention.fc_o.bias', 'fc1.weight', 'fc1.bias', 'fc2.weight', 'last_bn.running_mean', 'last_bn.running_var', 'fc2.bias', 'last_bn.weight', 'last_bn.bias']
    new_dict = {k:model_checkpoint[k] for k in keep_layer_name}
    new_model.load_state_dict(new_dict)

    torch.save(new_model.state_dict(), identity_layer)
    print("Save identity layer ckpt in ", identity_layer)

    # print ckpt
    # f = open("checkpoint_90.96.txt", "w")
    # # for key, value in state_dict["state_dict"].items(): # for multi-GPU trained model
    # for key, value in model_checkpoint.items():
    #     print(key, value.size(), sep="  ")
    #     # print(key, value, sep="  ")
    #     f.write("{key} {size} \n".format(key=key, size=value.size()))  # write into .txt


if __name__ == '__main__':
    face_finetune = '/home/xd/HUAWEI-CUP/mobilenetv3-master/pretrained/head_model_298_train2_0.871_test_0.956.pth' # 16.4 MB
    body_finetune = '/home/xd/HUAWEI-CUP/mobilenetv3-master/pretrained/body_model_277_train1_0.872_test_0.958.pth' # 16.4 MB
    # total_ckpt = '/home/xd/HUAWEI-CUP/mobilenetv3-master/pretrained/checkpoint-best_sgd.pth' # 80.0 MB
    total_ckpt = '/home/xd/HUAWEI-CUP/mobilenetv3-master/pretrained/checkpoint_90.96.pth' # 80.0 MB
    identity_layer = '/home/xd/HUAWEI-CUP/mobilenetv3-master/pretrained/identity_layer.pth' # 7.77 MB
    # identity_layer = '/home/xd/HUAWEI-CUP/mobilenetv3-master/pretrained/identity_layer_91.54.pth' # 11.2 MB for Simplified Self Attention
    main()

    # # compare different load mode
    # model_1 = Fusion_MobileNetV3_inference()
    # model_2 = Fusion_MobileNetV3_inference()
    #
    # # read checkpoints from 1 pth file
    # if total_ckpt is not None:
    #     model_checkpoint = torch.load(total_ckpt, map_location='cpu')
    #     model_checkpoint = model_checkpoint['model']
    #     print("Load model ckpt from %s" % total_ckpt)
    #     utils.load_state_dict(model_1, model_checkpoint, prefix='')
    #     # loaded_checkpoint = model.state_dict()
    #     # print(loaded_checkpoint)
    #
    # # read checkpoints from 3 pth file
    # if face_finetune and body_finetune:
    #     # load backbone
    #     face_checkpoint = torch.load(face_finetune, map_location='cpu')
    #     # face_checkpoint = face_checkpoint.state_dict()
    #     body_checkpoint = torch.load(body_finetune, map_location='cpu')
    #     # body_checkpoint = body_checkpoint.state_dict()
    #
    #     for name, param in list(face_checkpoint.items()):
    #         if "base_model.features" in name:
    #             new_name = name.replace("base_model.features", "face_backbone")
    #             face_checkpoint[new_name] = face_checkpoint.pop(name)
    #     utils.load_state_dict(model_2, face_checkpoint, prefix='')
    #
    #     for name, param in list(body_checkpoint.items()):
    #         if "base_model.features" in name:
    #             new_name = name.replace("base_model.features", "body_backbone")
    #             body_checkpoint[new_name] = body_checkpoint.pop(name)
    #     utils.load_state_dict(model_2, body_checkpoint, prefix='')
    #
    #     print(f"Load face ckpt from {face_finetune}\n"
    #           f"Load body ckpt from {body_finetune}")
    #
    # if identity_layer:
    #     # load identity layer
    #     identity_checkpoint = torch.load(identity_layer, map_location='cpu')
    #     # identity_checkpoint = identity_checkpoint.state_dict()
    #
    #     utils.load_state_dict(model_2, identity_checkpoint, prefix='')
    #     print(f"Load identity layer ckpt from {identity_layer}")
    #
    # dict_1 = model_1.state_dict()
    # dict_2 = model_2.state_dict()
    #
    # for key, value in dict_1.items():
    #     # print(key, value.size(), sep="  ")
    #     # print(key, value, sep="  ")
    #     # print(dict_1[key])
    #     # print(dict_2[key])
    #     if not dict_1[key].equal(dict_2[key]):
    #         print(f"{key} different! \n dict_1: {dict_1[key]} \n dict_2: {dict_2[key]}")



# *--------trainable params--------*
# fusion.attention.fc_o.weight
# fusion.attention.fc_o.bias
# fc1.weight
# fc1.bias
# fc2.weight
# fc2.bias
# last_bn.weight
# last_bn.bias
# classifier.weight
# classifier.bias

# *--------trainable params--------*
# fusion.attention.fc_q.weight
# fusion.attention.fc_q.bias
# fusion.attention.fc_k.weight
# fusion.attention.fc_k.bias
# fusion.attention.fc_v.weight
# fusion.attention.fc_v.bias
# fusion.attention.fc_o.weight
# fusion.attention.fc_o.bias
# fusion.attention.sr_conv.weight
# fusion.attention.sr_conv.bias
# fusion.attention.sr_ln.weight
# fusion.attention.sr_ln.bias
# fusion.attention.transform.conv.weight
# fusion.attention.transform.conv.bias
# fc1.weight
# fc1.bias
# fc2.weight
# fc2.bias
# last_bn.weight
# last_bn.bias
# classifier.weight
# classifier.bias