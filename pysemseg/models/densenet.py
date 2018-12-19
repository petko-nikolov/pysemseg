import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import checkpoint


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
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.kaiming_uniform_(self.conv.weight, nonlinearity='relu')


def _dense_layer_cp_factory(layer):
    def _dense_layer_fn(*features):
        concat_features = torch.cat(features, 1)
        return layer(concat_features)
    return _dense_layer_fn


class DenseBlock(nn.Module):
    def __init__(
            self, num_layers, num_input_features, growth_rate, drop_rate,
            efficient=False):
        super(DenseBlock, self).__init__()
        self.efficient = efficient
        self.layers = nn.ModuleList([
            DenseLayer(num_input_features + i * growth_rate, growth_rate, drop_rate)
            for i in range(num_layers)
        ])

    def forward(self, x):
        layer_outputs = []
        for layer in self.layers:
            if self.efficient:
                output = checkpoint.checkpoint(
                    _dense_layer_cp_factory(layer), x, *layer_outputs)
            else:
                output = layer(x)
                x = torch.cat([x, output], 1)
            layer_outputs.append(output)
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
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.kaiming_uniform_(self.conv.weight, nonlinearity='relu')


class TransitionUp(nn.Module):
    def __init__(self, n_input_features, n_output_features):
        super().__init__()
        self.conv = nn.ConvTranspose2d(
            n_input_features, n_output_features, kernel_size=3, stride=2,
            padding=1, output_padding=1)

    def reset_parameters(self):
        torch.nn.init.kaiming_uniform_(self.conv.weight, nonlinearity='relu')
        torch.nn.init.kaiming_uniform_(self.conv.bias, nonlinearity='relu')

    def forward(self, x):
        return self.conv(x)


class FCDenseNet(nn.Module):
    def __init__(
        self, in_channels, n_classes, growth_rate=32,
        n_init_features=48, drop_rate=0.2, blocks=(4, 5, 7, 10, 12, 15),
        efficient=False):
        super().__init__()

        self.efficient = efficient
        self.initial_conv = nn.Conv2d(
            in_channels, n_init_features, kernel_size=3, stride=1, padding=1)

        self.blocks_down = nn.ModuleList()
        self.transitions_down = nn.ModuleList()

        n_features = n_init_features
        self.skip_features = []

        for i, n_layers in enumerate(blocks[:-1]):
            self.blocks_down.append(
                DenseBlock(
                    n_layers, n_features, growth_rate, drop_rate,
                    efficient=self.efficient)
            )
            n_features = n_features + n_layers * growth_rate
            self.skip_features.append(n_features)
            self.transitions_down.append(
                TransitionDown(n_features, n_features, drop_rate)
            )

        self.blocks_down.append(
            DenseBlock(
                blocks[-1], n_features, growth_rate, drop_rate,
                efficient=self.efficient))
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

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.kaiming_uniform_(
            self.initial_conv.weight, nonlinearity='relu'
        )

    def _maybe_pad(self, x, size):
        hpad = max(size[0] - x.shape[2], 0)
        wpad = max(size[1] - x.shape[3], 0)
        if hpad + wpad > 0:
            lhpad = hpad // 2
            rhpad = hpad // 2 + hpad % 2
            lwpad = wpad // 2
            rwpad = wpad // 2 + wpad % 2

            x = F.pad(x, (lwpad, rwpad, lhpad, rhpad, 0, 0, 0, 0 ))
        return x



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
            x = self._maybe_pad(x, skip.shape[2:])
            x = torch.cat([x, skip], 1)
            x = block(x)

        x = self.score(x)
        return x


def fcdensenet56(in_channels, n_classes):
    return FCDenseNet(
        in_channels, n_classes, growth_rate=12, n_init_features=48,
        drop_rate=0.2, blocks=(4,) * 5, efficient=True
    )


def fcdensenet67(in_channels, n_classes):
    return FCDenseNet(
        in_channels, n_classes, growth_rate=16, n_init_features=48,
        drop_rate=0.2, blocks=(5,) * 5, efficient=True
    )


def fcdensenet103(in_channels, n_classes):
    return FCDenseNet(
        in_channels, n_classes, growth_rate=16, n_init_features=48,
        drop_rate=0.2, blocks=(4, 5, 7, 10, 12, 15),  efficient=True
    )
