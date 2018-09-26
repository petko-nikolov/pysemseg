import torch
import torch.nn as nn
import torch.nn.functional as F


class PaddingSameLayer2D(nn.Module):
    def __init__(self, conv):
        super().__init__()
        self.conv = conv

    def forward(self, x):
        x = F.pad(
            x,
            [self.conv.kernel_size[0]//2] * 2 +
            [self.conv.kernel_size[1]//2] * 2)
        return self.conv(x)


class DownLayer(nn.Module):
    def __init__(self, in_units, out_units):
        super().__init__()
        self.conv1 = PaddingSameLayer2D(
            nn.Conv2d(in_units, out_units, kernel_size=3))
        self.conv2 = PaddingSameLayer2D(
            nn.Conv2d(out_units, out_units, kernel_size=3))
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        if self.training:
            x = self.dropout(x)
        return x


class UpLayer(nn.Module):
    def __init__(self, in_units, out_units, upsample=True):
        super().__init__()
        self.conv1 = PaddingSameLayer2D(
            nn.Conv2d(in_units, out_units, kernel_size=3))
        self.conv2 = PaddingSameLayer2D(
            nn.Conv2d(out_units, out_units, kernel_size=3))
        self.upsample = upsample
        self.dropout = nn.Dropout(p=0.5)
        if self.upsample:
            self.conv3 = nn.ConvTranspose2d(
                    out_units, out_units // 2, kernel_size=2, stride=2)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        if self.training:
            x = self.dropout(x)
        if self.upsample:
            x = F.relu(self.conv3(x))
        return x


class UNet(nn.Module):
    def __init__(self, n_classes, in_channels):
        super().__init__()
        self.down_layers = [
            DownLayer(in_channels, 64),
            DownLayer(64, 128),
            DownLayer(128, 256),
            DownLayer(256, 512)
        ]
        self.interface1 = PaddingSameLayer2D(
            nn.Conv2d(512, 1024, kernel_size=3))
        self.interface2 = PaddingSameLayer2D(
            nn.Conv2d(1024, 512, kernel_size=3))
        self.interface_up = nn.ConvTranspose2d(
            512, 512, kernel_size=2, stride=2)
        self.up_layers = [
            UpLayer(1024, 512),
            UpLayer(512, 256),
            UpLayer(256, 128),
            UpLayer(128, 64, upsample=False)
        ]
        self.down_layers = nn.ModuleList(self.down_layers)
        self.up_layers = nn.ModuleList(self.up_layers)
        self.conv_classes = nn.Conv2d(64, n_classes, kernel_size=1)

    def forward(self, x):
        down_outputs = []
        for layer in self.down_layers:
            x = layer(x)
            down_outputs.append(x)
            x = F.max_pool2d(x, kernel_size=2)
        x = F.relu(self.interface1(x))
        x = F.relu(self.interface2(x))
        x = F.relu(self.interface_up(x))
        for i, layer in enumerate(self.up_layers):
            x = torch.cat([x, down_outputs[-(i + 1)]], dim=1)
            x = layer(x)
        x = self.conv_classes(x)
        return F.log_softmax(x, dim=1)
