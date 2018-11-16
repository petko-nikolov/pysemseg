import copy
import torch.nn as nn
import math


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
            'bottleneck0', Bottleneck(
                in_channels, channels, channels * expansion, stride,
                dilations[0], downsample=downsample))
        in_channels = channels * expansion
        for i in range(1, n_layers):
            self.add_module(
                'bottleneck{}'.format(i),
                Bottleneck(in_channels, channels, in_channels,
                           dilation=dilations[i]))


class ResNet(nn.Module):
    def __init__(self, in_channels, layers_config):
        super().__init__()
        assert len(layers_config) == 4
        channels = 64
        expansion = 4
        self.conv1 = nn.Conv2d(
            in_channels, channels, kernel_size=7, stride=2, padding=3,
            bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = ResBlock(
            channels, 64, expansion=expansion, **layers_config[0])
        channels = 64 * expansion
        self.layer2 = ResBlock(channels, 128, **layers_config[1])
        channels = 128 * expansion
        self.layer3 = ResBlock(channels, 256, **layers_config[2])
        channels = 256 * expansion
        self.layer4 = ResBlock(channels, 512, **layers_config[3])
        channels = 512 * channels

        self.reset_parameters()

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

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        return x


def resnet50(output_stride=16, additional_blocks=0, pretrained=False):
    rate  = 16 // output_stride
    stride = 1 if rate == 2 else 2
    blocks = [
        {'stride': 1, 'n_layers': 3},
        {'stride': 2, 'n_layers': 4},
        {'stride': stride, 'n_layers': 6, 'dilations': [rate] * 6},
        {'stride': 1, 'n_layers': 3,
         'dilations': [rate * ur for ur in [6, 12, 18]]}
    ]
    model = ResNet(3, blocks)
    return model


class DeepLabv3(nn.Module):
    def __init__(self, in_channels, n_classes, output_scale=16, multi_grid=(6, 12, 18)):
        assert in_channels == 3
        pass
