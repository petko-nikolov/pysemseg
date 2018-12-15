import torch
import torch.nn as nn
import torch.nn.functional as F
from pysemseg.models.resnet import resnet50, resnet101


def _maybe_pad(x, size):
    hpad = size[0] - x.shape[2]
    wpad = size[1] - x.shape[3]
    if hpad + wpad > 0:
        x = F.pad(x, (0, wpad, 0, hpad, 0, 0, 0, 0 ))
    return x


class DownLayer(nn.Module):
    def __init__(self, in_units, out_units, batch_norm=True):
        super().__init__()
        self.batch_norm = batch_norm
        self.conv1 = nn.Conv2d(in_units, out_units, kernel_size=3, padding=1)
        if batch_norm:
            self.bn1 = nn.BatchNorm2d(out_units)
        self.conv2 = nn.Conv2d(out_units, out_units, kernel_size=3, padding=1)
        if batch_norm:
            self.bn2 = nn.BatchNorm2d(out_units)
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        if self.batch_norm:
            x = self.bn1(x)
        x = F.relu(self.conv2(x))
        if self.batch_norm:
            x = self.bn2(x)
        if self.training:
            x = self.dropout(x)
        return x


class UpLayer(nn.Module):
    def __init__(self, in_units, out_units, upsample=True, batch_norm=True):
        super().__init__()
        self.batch_norm = batch_norm
        self.conv1 = nn.Conv2d(in_units, out_units, kernel_size=3, padding=1)
        if batch_norm:
            self.bn1 = nn.BatchNorm2d(out_units)
        self.conv2 = nn.Conv2d(out_units, out_units, kernel_size=3, padding=1)
        if batch_norm:
            self.bn2 = nn.BatchNorm2d(out_units)
        self.upsample = upsample
        self.dropout = nn.Dropout(p=0.5)
        if self.upsample:
            self.conv3 = nn.ConvTranspose2d(
                out_units, out_units // 2, kernel_size=2, stride=2,
                padding=0, output_padding=0)
            if self.batch_norm:
                self.bn3 = nn.BatchNorm2d(out_units // 2)


    def forward(self, x):
        x = F.relu(self.conv1(x))
        if self.batch_norm:
            x = self.bn1(x)
        x = F.relu(self.conv2(x))
        if self.batch_norm:
            x = self.bn2(x)
        if self.training:
            x = self.dropout(x)
        if self.upsample:
            x = F.relu(self.conv3(x))
            if self.batch_norm:
                x = self.bn3(x)
        return x


class UNet(nn.Module):
    def __init__(self, n_classes, in_channels, batch_norm=True):
        super().__init__()
        self.down_layers = [
            DownLayer(in_channels, 64, batch_norm),
            DownLayer(64, 128, batch_norm),
            DownLayer(128, 256, batch_norm),
            DownLayer(256, 512, batch_norm)
        ]
        self.interface1 = nn.Conv2d(512, 1024, kernel_size=3, padding=1)
        self.interface2 = nn.Conv2d(1024, 512, kernel_size=3, padding=1)
        self.interface_up = nn.ConvTranspose2d(
            512, 512, kernel_size=2, stride=2)
        self.up_layers = [
            UpLayer(1024, 512, batch_norm=batch_norm),
            UpLayer(512, 256, batch_norm=batch_norm),
            UpLayer(256, 128, batch_norm=batch_norm),
            UpLayer(128, 64, upsample=False, batch_norm=batch_norm)
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
            x = _maybe_pad(x, down_outputs[-(i + 1)].shape[2:])
            x = torch.cat([x, down_outputs[-(i + 1)]], dim=1)
            x = layer(x)
        x = self.conv_classes(x)
        return x


class UNetResNetV1(nn.Module):
    def __init__(
            self, n_classes, network, skip_channels, interface_channels,
            up_channels):
        super().__init__()
        self.network = network
        self.bottleneck_channels = interface_channels // self.network.expansion
        self.interface = UpLayer(interface_channels, self.bottleneck_channels)
        self.up_layers = nn.ModuleList([
            UpLayer(skip_channels[-1] + self.bottleneck_channels, up_channels[0]),
            UpLayer(skip_channels[-2] + up_channels[0], up_channels[1]),
            UpLayer(skip_channels[-3] + up_channels[1], up_channels[2]),
            UpLayer(skip_channels[-4] + up_channels[2], up_channels[3])
        ])
        self.downsample = nn.ModuleList([
            nn.Conv2d(
                self.network.expansion * skip_channels[0], skip_channels[0], 1
            ),
            nn.Conv2d(
                self.network.expansion * skip_channels[1], skip_channels[1], 1
            ),
            nn.Conv2d(
                self.network.expansion * skip_channels[2], skip_channels[2], 1
            ),
            nn.Conv2d(
                self.network.expansion * skip_channels[3], skip_channels[3], 1
            ),
        ])

        self.conv_classes = nn.Conv2d(64, n_classes, kernel_size=1)

    def _resnet_forward(self, x):
        skips = []
        x = self.network.conv1(x)
        skips.append(x)
        x = self.network.bn1(x)
        x = self.network.relu(x)
        x = self.network.maxpool(x)
        skips.append(x)
        block_outputs = []
        for i, layer in enumerate(self.network.layers):
            height, width = x.shape[2:]
            x = layer(x)
            if (x.shape[2] < height and x.shape[3] < width and
                i < len(self.network.layers) - 1):
                skips.append(x)
        return x, skips

    def forward(self, x):
        input_tensor = x
        x, skips = self._resnet_forward(x)
        x = self.interface(x)
        for i in range(1, len(self.up_layers)):
            skip = self.downsample[-(i + 1)](skips[-(i + 1)])
            x = torch.cat([x, skip], dim=1)
            x = _maybe_pad(x, skip.shape[2:])
            x = self.up_layer[i](x)
        x = torch.cat([x, input_tensor], dim=1)
        x = _maybe_pad(x, input_tensor.shape[2:])
        scores = self.conv_classes(x)
        return x

def unet_resnet101(in_channels, n_classes):
    net = resnet101(in_channels=in_channels, output_stride=32)
    return UNetResNetV1(
        n_classes=n_classes, network=net,
        skip_channels=[64, 64, 128, 256],
        interface_channels=2048,
        up_channels=[512, 256, 128, 64]
        )
