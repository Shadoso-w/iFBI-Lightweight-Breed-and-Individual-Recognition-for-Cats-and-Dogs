import torch.nn as nn
import time
import torch
from torch.nn import init

## Classifier for Identification
class Identity_Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(2048, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 1)
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

    def maxmin_norm(self, x):
        a = torch.min(x)
        b = torch.max(x)
        normalized_x = (x - a) / (b - a)
        return normalized_x

    def forward(self, x, y):
        out = torch.cat([x, y], dim=1)
        out = self.fc1(out)
        out = nn.LeakyReLU()(out)
        out = self.fc2(out)
        out = nn.LeakyReLU()(out)
        out = self.fc3(out)
        out = self.maxmin_norm(out)
        return out

class Cosine_Classifier(nn.Module):
    def __init__(self, similar_threshold):
        super().__init__()
        self.similar_threshold = similar_threshold

    def forward(self, x, y):
        out = torch.cosine_similarity(x, y, dim=1, eps=1e-8)
        out[out >= self.similar_threshold] = 1
        out[out < self.similar_threshold] = 0
        return out


if __name__ == '__main__':
    model = Identity_Classifier()
    feature_1 = torch.rand(180, 1024)
    feature_2 = torch.rand(180, 1024)
    all_start = time.time()
    output = model(feature_1, feature_2)
    all_end = time.time()
    print("spent time: ", all_end - all_start)
    print("temp test")