import math
import re
from collections import OrderedDict
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo


RESNET_CKPT = {
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth'
}


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
        expansion = 4
        channels = layers_config[0]['channels']
        self.conv1 = nn.Conv2d(
            in_channels, channels, kernel_size=7, stride=2, padding=3,
            bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layers = nn.ModuleList()
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
        block_outputs = []
        for layer in self.layers:
            x = layer(x)
            block_outputs.append(x)
        return x, block_outputs


def resnet(layers, pretrained_model=None, in_channels=3, output_stride=16,
            multi_grid=(1, 2, 4), add_blocks=0):

    assert output_stride in [8, 16, 32]
    if output_stride == 8:
        rate3 = 2 * layers[2]
        stride3 = 1
        rate4 = [rate3 * 2 * mg for mg in multi_grid]
        stride4 = 1
    elif output_stride == 16:
        rate3 = 1 * layers[2]
        stride3 = 2
        rate4 = [rate3 * 2 * mg for mg in multi_grid]
        stride4 = 1
    elif output_stride == 32:
        rate3 = 1 * layers[2]
        stride3 = 2
        rate4 = 1 * layers[3]
        stride4 = 2

    blocks = [
        {
            'stride': 1,
            'n_layers': layers[0],
            'channels': 64
        },
        {
            'stride': 2,
            'n_layers': layers[1],
            'channels': 128
        },
        {
            'stride': stride3,
            'n_layers': layers[2],
            'dilations': rate3,
            'channels': 256
        },
        {
            'stride': stride4,
            'n_layers': layers[3],
            'dilations': rate4,
            'channels': 512
        }
    ]
    rate = rate3[0] * 4
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
    resnet = ResNet(in_channels, blocks)
    if pretrained_model is not None:
        resnet.load_pretrained_model(pretrained_model)

    return resnet


def resnet50(pretrained=True, **kwargs):
    return resnet(
        [3, 4, 6, 3],
        pretrained_model=RESNET_CKPT['resnet50'] if pretrained else None,
        **kwargs
    )


def resnet101(pretrained=True, **kwargs):
    return resnet(
        [3, 4, 23, 3],
        pretrained_model=RESNET_CKPT['resnet101'] if pretrained else None,
        **kwargs
    )


def resnet152(pretrained=True, **kwargs):
    return resnet(
        [3, 8, 36, 3],
        pretrained_model=RESNET_CKPT['resnet152'] if pretrained else None
        **kwargs
    )
