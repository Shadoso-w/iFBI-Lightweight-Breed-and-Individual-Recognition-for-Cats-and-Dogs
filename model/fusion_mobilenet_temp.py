'''MobileNetV3 in PyTorch.

See the paper "Inverted Residuals and Linear Bottlenecks:
Mobile Networks for Classification, Detection and Segmentation" for more details.
'''
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init

from torchvision.models.mobilenetv3 import mobilenet_v3_small, mobilenet_v3_large
from .mobiledynamic import DynamicMobileNetV3Large

import time

class AdaIN(nn.Module):
    def __init__(self, alpha=0.3):
        super().__init__()
        self.alpha = alpha # control the degree of transform

        assert 0 <= self.alpha <= 1

    def calc_mean_std(self, feat, eps=1e-5):
        # eps is a small value added to the variance to avoid divide-by-zero.
        size = feat.size()
        assert (len(size) == 4)
        N, C = size[:2]
        feat_var = feat.view(N, C, -1).var(dim=2) + eps
        feat_std = feat_var.sqrt().view(N, C, 1, 1)
        feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
        return feat_mean, feat_std

    def adaptive_instance_normalization(self, content_feat, style_feat):
        assert (content_feat.size()[:2] == style_feat.size()[:2])
        size = content_feat.size()
        style_mean, style_std = self.calc_mean_std(style_feat)
        content_mean, content_std = self.calc_mean_std(content_feat)

        normalized_feat = (content_feat - content_mean.expand(
            size)) / content_std.expand(size)
        return normalized_feat * style_std.expand(size) + style_mean.expand(size)

    def forward(self, content_feat, style_feats):
        t = self.adaptive_instance_normalization(content_feat, style_feats)
        t = self.alpha * t + (1 - self.alpha) * content_feat
        return t

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
            self.attention = EMSA(d_model=960, d_k=960, d_v=960,  h=8, H=8, W=8, ratio=2, apply_transform=True)
        elif mode == 'MUSE Attention':
            from fightingcv_attention.attention.MUSEAttention import MUSEAttention
            self.attention = MUSEAttention(d_model=960, d_k=960, d_v=960,  h=8)
        elif mode == 'UFO Attention':
            from fightingcv_attention.attention.UFOAttention import UFOAttention
            self.attention = UFOAttention(d_model=960, d_k=960, d_v=960,  h=8)
        else:
            print("Please choose supported attention")

    ## todo: new fusion module: add

    def img2seq(self, img):
        bs, C, H, W = img.shape
        seq = img.view(bs, C, H*W)
        seq = seq.transpose(1, 2)
        return seq

    def seq2img(self, seq):
        bs, N, C = seq.shape
        H = W = int(np.sqrt(N))
        seq = seq.transpose(1, 2)
        img = seq.view(bs, C, H, W)
        return img

    def forward(self, x, y):
        query = self.img2seq(x)
        key = self.img2seq(x)
        valve = self.img2seq(y)

        out = self.attention(query, key, valve)
        out = self.seq2img(out)
        return out

class Fusion_MobileNetV3(nn.Module):
    def __init__(self, num_classes=1000, dropout_keep_prob=0.5, embedding_size=1024, alpha=0.5):
        super(Fusion_MobileNetV3, self).__init__()

        # face backbone
        # self.face_backbone = mobilenet_v3_large().features
        self.face_backbone = DynamicMobileNetV3Large().base_model.features

        # body backbone
        self.body_backbone = DynamicMobileNetV3Large().base_model.features

        # fusion module
        self.fusion = AdaIN(alpha)
        # self.fusion = CrossAttention(mode='UFO Attention')

        # identity layer
        self.avg = nn.AdaptiveAvgPool2d((1, 1))
        self.Dropout = nn.Dropout(1 - dropout_keep_prob)
        self.fc1 = nn.Linear(960, embedding_size)
        self.fc2 = nn.Linear(embedding_size, embedding_size)
        self.last_bn = nn.BatchNorm1d(embedding_size, eps=0.001, momentum=0.1, affine=True)

        # classifier for Cross-Entropy Loss
        if self.training == True:
            self.classifier = nn.Linear(embedding_size, num_classes)

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

    def forward(self, face, body):
        # feature extract backbone
        if body.any() and face.any():
            body = self.body_backbone(body) # Tensor(64, 960, 7, 7)
            face = self.face_backbone(face)
        elif body.any():
            body = self.body_backbone(body)
            face = body
        elif face.any():
            face = self.face_backbone(face)
            body = face
        else:
            print("No Any Input!")
            exit(-1)

        # fusion module
        out = self.fusion(face, body) # Tensor(64, 960, 7, 7)

        # identity layer
        if self.training == False:
            out = self.avg(out)
            out = out.view(out.size(0), -1)  # Tensor(64, 960)
            # out = self.Dropout(out)
            out = self.fc1(out)
            out = self.fc2(out)
            out = self.last_bn(out)
            out = F.normalize(out, p=2, dim=1)
            return out #  Tensor(180, 1024)
        out = self.avg(out)
        out = out.view(out.size(0), -1)  # Tensor(64, 960)
        out = self.Dropout(out)
        out = self.fc1(out)
        out = self.fc2(out)
        before_normalize = self.last_bn(out)
        out = F.normalize(before_normalize, p=2, dim=1)
        cls = self.classifier(before_normalize)
        return out, cls


class Fusion_MobileNetV3_inference(nn.Module):
    def __init__(self, num_classes=1000, dropout_keep_prob=0.5, embedding_size=1024, alpha=0.5):
        super(Fusion_MobileNetV3_inference, self).__init__()

        # face backbone
        # self.face_backbone = mobilenet_v3_large().features
        self.face_backbone = DynamicMobileNetV3Large().base_model.features

        # body backbone
        self.body_backbone = DynamicMobileNetV3Large().base_model.features

        # fusion module
        self.fusion = AdaIN(alpha)
        # self.fusion = CrossAttention(mode='Self Attention')

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

    def forward(self, face, body):
        # feature extract backbone
        if body.any() and face.any():
            body = self.body_backbone(body)  # Tensor(64, 960, 7, 7)
            face = self.face_backbone(face)
        elif body.any():
            body = self.body_backbone(body)
            face = body
        elif face.any():
            face = self.face_backbone(face)
            body = face
        else:
            print("No Any Input!")
            exit(-1)

        # fusion module
        out = self.fusion(face, body) # Tensor(64, 960, 7, 7)

        # identity layer
        out = self.avg(out)
        out = out.view(out.size(0), -1)  # Tensor(64, 960)
        # out = self.Dropout(out)
        out = self.fc1(out)
        out = self.fc2(out)
        out = self.last_bn(out)
        out = F.normalize(out, p=2, dim=1)
        return out #  Tensor(180, 1024)



if __name__ == '__main__':
    model = Fusion_MobileNetV3()
    seed = 0
    face_input = torch.rand(180, 3, 224, 224)
    # face_input = torch.zeros(180, 3, 224, 224)
    body_input = torch.rand(180, 3, 224, 224)
    # body_input = torch.zeros(180, 3, 224, 224)
    all_start = time.time()
    output = model(face_input, body_input)
    all_end = time.time()
    print("spent time: ", all_end - all_start)
    print("temp test")
