import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import mobilenet_v3_large, MobileNet_V3_Large_Weights

class Attention(nn.Module):
    def __init__(self, in_planes, ratio, K, temprature=30, init_weight=True):
        super().__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.temprature = temprature
        self.use_hidden_layer = in_planes > ratio
        if self.use_hidden_layer:
            hidden_planes = in_planes // ratio
            self.net = nn.Sequential(
                nn.Conv2d(in_planes, hidden_planes, kernel_size=1, bias=False),
                nn.ReLU(),
                nn.Conv2d(hidden_planes, K, kernel_size=1, bias=False)
            )
        else:
            self.net = nn.Conv2d(in_planes, K, kernel_size=1, bias=False)

        if init_weight:
            self._initialize_weights()

    def update_temprature(self):
        if self.temprature > 1:
            self.temprature -= 1

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        att = self.avgpool(x)  # bs, dim, 1, 1
        att = self.net(att).view(x.shape[0], -1)  # bs, K
        return F.softmax(att / self.temprature, -1)

class DynamicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride, padding=0, dilation=1, groups=1, bias=True, K=4, temprature=30, ratio=4, init_weight=True):
        super().__init__()
        self.in_planes = in_planes
        self.out_planes = out_planes
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.bias = bias
        self.K = K
        self.init_weight = init_weight
        self.attention = Attention(in_planes=in_planes, ratio=ratio, K=K, temprature=temprature, init_weight=init_weight)

        self.weight = nn.Parameter(torch.randn(K, out_planes, in_planes // groups, kernel_size, kernel_size), requires_grad=True)
        if bias:
            self.bias = nn.Parameter(torch.randn(K, out_planes), requires_grad=True)
        else:
            self.bias = None

        if self.init_weight:
            self._initialize_weights()

    def _initialize_weights(self):
        for i in range(self.K):
            nn.init.kaiming_uniform_(self.weight[i])

    def forward(self, x):
        bs, in_planes, h, w = x.shape
        softmax_att = self.attention(x)  # bs, K
        x = x.view(1, -1, h, w)
        weight = self.weight.view(self.K, -1)  # K, -1
        aggregate_weight = torch.mm(softmax_att, weight).view(bs * self.out_planes, self.in_planes // self.groups, self.kernel_size, self.kernel_size)  # bs*out_p, in_p, k, k

        if self.bias is not None:
            bias = self.bias.view(self.K, -1)  # K, out_p
            aggregate_bias = torch.mm(softmax_att, bias).view(-1)  # bs, out_p
            output = F.conv2d(x, weight=aggregate_weight, bias=aggregate_bias, stride=self.stride, padding=self.padding, groups=self.groups * bs, dilation=self.dilation)
        else:
            output = F.conv2d(x, weight=aggregate_weight, bias=None, stride=self.stride, padding=self.padding, groups=self.groups * bs, dilation=self.dilation)

        out_h = (h + 2 * self.padding - self.dilation * (self.kernel_size - 1) - 1) // self.stride + 1
        out_w = (w + 2 * self.padding - self.dilation * (self.kernel_size - 1) - 1) // self.stride + 1

        output = output.view(bs, self.out_planes, out_h, out_w)
        return output

class DynamicMobileNetV3Large(nn.Module):
    def __init__(self, num_classes=1000, dynamic_layers=[3, 6, 10, 15], K=4, temperature=30):
        super(DynamicMobileNetV3Large, self).__init__()
        self.base_model = mobilenet_v3_large(weights=MobileNet_V3_Large_Weights.IMAGENET1K_V1)
        self.num_classes = num_classes
        self.dynamic_layers = dynamic_layers
        self.K = K
        self.temperature = temperature
        self.base_model.classifier[3] = nn.Linear(in_features=1280, out_features=num_classes, bias=True)

        # # 替换指定的层为DynamicConv
        # for name, module in self.base_model.named_modules():
        #     if isinstance(module, nn.Conv2d):
        #         if name in self.dynamic_layers:
        #             in_planes = module.in_channels
        #             out_planes = module.out_channels
        #             kernel_size = module.kernel_size[0]  # 假设方形卷积核
        #             stride = module.stride[0]  # 假设各方向步幅相等
        #             padding = module.padding[0]  # 假设各方向填充相等
        #             groups = module.groups
        #             bias = module.bias is not None

        #             # 检查 in_planes 是否大于 ratio
        #             dynamic_conv = DynamicConv(
        #                 in_planes=in_planes, out_planes=out_planes, kernel_size=kernel_size,
        #                 stride=stride, padding=padding, groups=groups, bias=bias, K=K, temprature=self.temperature
        #             )

        #             # 获取父模块
        #             parent_name = '.'.join(name.split('.')[:-1])
        #             parent_module = dict(self.base_model.named_modules())[parent_name]
        #             setattr(parent_module, name.split('.')[-1], dynamic_conv)
    def _modify_dynamic_layers(self):
        for i in self.dynamic_layers:
            layer = self.base_model.features[i]
            if isinstance(layer, nn.Sequential):
                for j, sublayer in enumerate(layer):
                    if isinstance(sublayer, nn.Conv2d):
                        in_planes = sublayer.in_channels
                        out_planes = sublayer.out_channels
                        kernel_size = sublayer.kernel_size[0]
                        stride = sublayer.stride[0]
                        padding = sublayer.padding[0]
                        dilation = sublayer.dilation[0]
                        groups = sublayer.groups
                        bias = sublayer.bias is not None
                        
                        dynamic_conv = DynamicConv(
                            in_planes=in_planes, 
                            out_planes=out_planes, 
                            kernel_size=kernel_size, 
                            stride=stride, 
                            padding=padding, 
                            dilation=dilation, 
                            groups=groups, 
                            bias=bias, 
                            K=self.K, 
                            temperature=self.temperature
                        )
                        layer[j] = dynamic_conv


    def forward(self, x):
        return self.base_model(x)

if __name__ == '__main__':
    # 定义要替换为DynamicConv的层
    dynamic_layers = [
        'features.0.0',           # 第1个卷积层
        'features.3.block.0.0',   # 第1个InvertedResidual Block的第1个卷积层
        'features.6.block.0.0',   # 第2个InvertedResidual Block的第1个卷积层
        'features.12.block.0.0',  # 第3个InvertedResidual Block的第1个卷积层
        'features.15.block.0.0'   # 第4个InvertedResidual Block的第1个卷积层
    ]

    # 初始化模型
    model = DynamicMobileNetV3Large(num_classes=38, dynamic_layers=dynamic_layers)

    # 用随机输入测试修改后的模型
    input = torch.randn(2, 3, 224, 224)
    print(model)
    output = model(input)
    print(output.shape)
