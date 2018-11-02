import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from models.layers import PaddingSameConv2d


class DenseLayer(nn.Sequential):
    def __init__(self, n_input_features, growth_rate, drop_rate):
        super().__init__()
        self.add_module('norm', nn.BatchNorm2d(n_input_features))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module(
            'conv',
            nn.Conv2d(
                n_input_features, growth_rate, kernel_size=3, stride=1,
                padding=1))
        self.add_module('drop', nn.Dropout(drop_rate))


class DenseBlock(nn.Module):
    def __init__(self, num_layers, num_input_features, growth_rate, drop_rate):
        super(DenseBlock, self).__init__()
        self.layers = nn.ModuleList([
            DenseLayer(num_input_features + i * growth_rate, growth_rate, drop_rate)
            for i in range(num_layers)
        ])

    def forward(self, x):
        layer_outputs = []
        for layer in self.layers:
            c = layer(x)
            layer_outputs.append(c)
            x = torch.cat([x, c], 1)
        return torch.cat(layer_outputs, 1)


class TransitionDown(nn.Sequential):
    def __init__(self, n_input_features, n_output_features, drop_rate=0.2):
        super().__init__()
        self.add_module('norm', nn.BatchNorm2d(n_input_features))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', nn.Conv2d(
            n_input_features, n_output_features, kernel_size=1, stride=1))
        self.add_module('drop', nn.Dropout(drop_rate))
        self.add_module('pool', nn.MaxPool2d(kernel_size=2, stride=2))


class TransitionUp(nn.Module):
    def __init__(self, n_input_features, n_output_features):
        super().__init__()
        self.conv = nn.ConvTranspose2d(
            n_input_features, n_output_features, kernel_size=3, stride=2,
            padding=1, output_padding=1)

    def forward(self, x):
        return self.conv(x)


class FCDenseNet(nn.Module):
    def __init__(
        self, in_channels, n_classes, growth_rate=32,
        n_init_features=48, drop_rate=0.2, blocks=(4, 5, 7, 10, 12, 15)):
        super().__init__()
        self.initial_conv = nn.Conv2d(
            in_channels, n_init_features, kernel_size=3, stride=1, padding=1)

        self.blocks_down = nn.ModuleList()
        self.transitions_down = nn.ModuleList()

        n_features = n_init_features
        self.skip_features = []

        for i, n_layers in enumerate(blocks[:-1]):
            self.blocks_down.append(
                DenseBlock(n_layers, n_features, growth_rate, drop_rate)
            )
            n_features = n_features + n_layers * growth_rate
            self.skip_features.append(n_features)
            self.transitions_down.append(
                TransitionDown( n_features, n_features, drop_rate)
            )

        self.blocks_down.append(
            DenseBlock(blocks[-1], n_features, growth_rate, drop_rate))
        n_features = blocks[-1] * growth_rate

        self.blocks_up = nn.ModuleList()
        self.transitions_up = nn.ModuleList()

        for i in range(len(blocks) - 2, -1, -1):
            self.transitions_up.append(TransitionUp(n_features, n_features))
            n_features = n_features + self.skip_features[i]
            self.blocks_up.append(
                DenseBlock(blocks[i], n_features, growth_rate, drop_rate)
            )
            n_features = growth_rate * blocks[i]

        self.score = nn.Conv2d(n_features, n_classes, kernel_size=1, stride=1)

    def forward(self, x):
        x = self.initial_conv(x)
        skips = []
        for block, trans in zip(self.blocks_down, self.transitions_down):
            c = block(x)
            x = torch.cat([c, x], 1)
            skips.append(x)
            x = trans(x)

        x = self.blocks_down[-1](x)
        skips = skips[::-1]

        for block, trans, skip in zip(
                self.blocks_up, self.transitions_up, skips):
            x = trans(x)
            x = torch.cat([x, skip], 1)
            x = block(x)

        x = self.score(x)

        return x


class FCDenseNet56(FCDenseNet):
    def __init__(self, in_channels, n_classes):
        super().__init__(
            in_channels, n_classes, growth_rate=12,
            n_init_features=48, drop_rate=0.2, blocks=(4, 4, 4, 4, 4, 4)
        )


class FCDenseNet67(FCDenseNet):
    def __init__(self, in_channels, n_classes):
        super().__init__(
            in_channels, n_classes, growth_rate=16,
            n_init_features=48, drop_rate=0.2, blocks=(5, 5, 5, 5, 5, 5)
        )


class FCDenseNet103(FCDenseNet):
    def __init__(self, in_channels, n_classes):
        super().__init__(
            in_channels, n_classes, growth_rate=16,
            n_init_features=48, drop_rate=0.2, blocks=(4, 5, 7, 10, 12, 15)
        )
