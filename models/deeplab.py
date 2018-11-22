import math
import re
from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo


class Bottleneck(nn.Module):
    def __init__(self, in_channels, bottleneck_channels, out_channels,
        stride=1, dilation=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels, bottleneck_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(bottleneck_channels)
        self.conv2 = nn.Conv2d(
            bottleneck_channels, bottleneck_channels, kernel_size=3,
            stride=stride, padding=(dilation, dilation),
            dilation=dilation, bias=False)
        self.bn2 = nn.BatchNorm2d(bottleneck_channels)
        self.conv3 = nn.Conv2d(
            bottleneck_channels, out_channels, kernel_size=1,
            bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResBlock(nn.Sequential):
    def __init__(self, in_channels, channels, n_layers, dilations=None,
            stride=1, expansion=4):
        super().__init__()
        if dilations is None:
            dilations = [1] * n_layers
        downsample = None
        if stride != 1 or in_channels != channels * expansion:
            downsample = nn.Sequential(
                nn.Conv2d(in_channels, channels * expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(channels * expansion),
            )

        layers = []
        self.add_module(
            '0', Bottleneck(
                in_channels, channels, channels * expansion, stride,
                dilations[0], downsample=downsample))
        in_channels = channels * expansion
        for i in range(1, n_layers):
            self.add_module(
                '{}'.format(i),
                Bottleneck(in_channels, channels, in_channels,
                           dilation=dilations[i]))


class ResNet(nn.Module):
    def __init__(self, in_channels, layers_config):
        super().__init__()
        assert len(layers_config) == 4
        expansion = 4
        channels = layers_config[0]['channels']
        self.conv1 = nn.Conv2d(
            in_channels, channels, kernel_size=7, stride=2, padding=3,
            bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layers = nn.Sequential()
        for i, layer_config in enumerate(layers_config):
            self.layers.add_module(
                'layer{}'.format(i + 1),
                ResBlock(channels, **layer_config, expansion=expansion)
            )
            channels = layer_config['channels'] * expansion

        self.reset_parameters()

    def load_pretrained_model(self, url):
        def replace_key(key):
            return re.sub(r'layer(\d)', r'layers.layer\g<1>', key)
        state_dict = model_zoo.load_url(url)
        state_dict = OrderedDict([
            (replace_key(key), value) for key, value in state_dict.items()
        ])
        self.load_state_dict(state_dict, strict=False)

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layers(x)
        return x


def resnet50(output_stride=16, pretrained=False, multi_grid=(1,2,4),
             add_blocks=0):
    assert output_stride in [8, 16]
    rate3  = 16 // output_stride
    stride3 = 1 if rate3 == 2 else 2
    blocks = [
        {
            'stride': 1,
            'n_layers': 3,
            'channels': 64
        },
        {
            'stride': 2,
            'n_layers': 4,
            'channels': 128
        },
        {
            'stride': stride3,
            'n_layers': 6,
            'dilations': [rate3] * 6,
            'channels': 256
        },
        {
            'stride': 1,
            'n_layers': 3,
            'dilations': [rate3 * 2 * mg for mg in multi_grid],
            'channels': 512
        }
    ]
    rate = rate3 * 4
    for _ in range(add_blocks):
        dilations = [rate * mg for mg in multi_grid]
        blocks.append(
            {
                'stride': 1,
                'n_layers': 3,
                'dilations': dilations,
                'channels': 512
            }
        )

    resnet = ResNet(3, blocks)

    if pretrained:
        resnet.load_pretrained_model(
            'https://download.pytorch.org/models/resnet50-19c8e357.pth')

    return resnet

def resnet101(output_stride=16, pretrained=False, multi_grid=(1,2,4)):
    assert output_stride in [8, 16]
    rate3  = 16 // output_stride
    stride3 = 1 if rate3 == 2 else 2
    blocks = [
        {
            'stride': 1,
            'n_layers': 3,
            'channels': 64
        },
        {
            'stride': 2,
            'n_layers': 4,
            'channels': 128
        },
        {
            'stride': stride3,
            'n_layers': 23,
            'dilations': [rate3] * 6,
            'channels': 256
        },
        {
            'stride': 1,
            'n_layers': 3,
            'dilations': [rate3 * 2 * mg for mg in multi_grid],
            'channels': 512
        }
    ]
    rate = rate3 * 4
    for _ in range(add_blocks):
        dilations = [rate * mg for mg in multi_grid]
        blocks.append(
            {
                'stride': 1,
                'n_layers': 3,
                'dilations': dilations,
                'channels': 512
            }
        )
    return ResNet(3, blocks)


class ASPPModule(nn.Module):
    def __init__(self, in_channels, out_channels, rates):
        super().__init__()
        assert len(rates) == 3
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1,
                               bias=False)
        self.conv2 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, dilation=rates[0],
            padding=rates[0], bias=False)
        self.conv3 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, dilation=rates[1],
            padding=rates[1], bias=False)
        self.conv4 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, dilation=rates[2],
            padding=rates[2], bias=False)
        self.global_avg_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        )
        self.bn = nn.BatchNorm2d(5 * out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.final_conv = nn.Conv2d(
            5 * out_channels, out_channels, kernel_size=1, bias=False)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x3 = self.conv3(x)
        x4 = self.conv4(x)
        x5 = self.global_avg_pool(x)
        x5 = F.interpolate(x5, size=x1.size()[2:], mode='bilinear')
        x = torch.cat([x1, x2, x3, x4, x5], dim=1)
        x = self.relu(self.bn(x))
        x = self.final_conv(x)
        return x

class DeepLabV3(nn.Module):
    def __init__(self, channels, n_classes, backbone_model, aspp_module,
                 finetune_bn):
        super().__init__()
        self.backbone = backbone_model
        self.aspp = aspp_module
        self.finetune_bn = finetune_bn
        self.score = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, n_classes, kernel_size=1)
        )

    def forward(self, x):
        input_size = x.shape[2:]
        x = self.backbone(x)
        x = self.aspp(x)
        x = F.interpolate(x, size=input_size, mode='bilinear')
        x = self.score(x)
        return x

    def train(self, mode=True):
        super().train(mode)
        if not self.finetune_bn:
            for module in self.modules():
                if isinstance(module, nn.BatchNorm2d):
                    module.train(False)


class DeepLabV3ResNet50(DeepLabV3):
    def __init__(self, in_channels, n_classes, output_stride=16,
                 aspp_rates=[6, 12, 18], multi_grid=[1, 2, 4],
                 finetune_bn=True):
        assert in_channels == 3
        resnet = resnet50(
            output_stride=output_stride, pretrained=True,
            multi_grid=multi_grid)
        rate = 2 if output_stride == 8 else 1
        aspp = ASPPModule(2048, 256, [r * rate for r in aspp_rates])
        super().__init__(256, n_classes, resnet, aspp, finetune_bn)


class DeepLabV3Reaspp_ratessNet101(DeepLabV3):
    def __init__(self, in_channels, n_classes, output_stride=16,
                 aspp_rates=[6, 12, 18], multi_grid=[1, 2, 4],
                 finetune_bn=True):
        assert in_channels == 3
        resnet = resnet101(
            output_stride=output_stride, pretrained=True,
            multi_grid=multi_grid)
        aspp = ASPPModule(2048, 256, [r * rate for r in aspp_rates])
        super().__init__(256, n_classes, resnet, aspp, finetune_bn)
