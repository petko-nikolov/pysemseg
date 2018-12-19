import torch
import torch.nn as nn
import torch.nn.functional as F
from pysemseg.models.resnet import resnet50, resnet101, resnet152, ResBlock


def _maybe_pad(x, size):
    hpad = max(size[0] - x.shape[2], 0)
    wpad = max(size[1] - x.shape[3], 0)
    if hpad + wpad > 0:
        lhpad = hpad // 2
        rhpad = hpad // 2 + hpad % 2
        lwpad = wpad // 2
        rwpad = wpad // 2 + wpad % 2

        x = F.pad(x, (lwpad, rwpad, lhpad, rhpad, 0, 0, 0, 0 ))
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
        self.dropout = nn.Dropout(p=0.0)

    def forward(self, x):
        x = F.relu(self.conv1(x), inplace=True)
        if self.batch_norm:
            x = self.bn1(x)
        x = F.relu(self.conv2(x), inplace=True)
        if self.batch_norm:
            x = self.bn2(x)
        if self.training:
            x = self.dropout(x)
        return x


class UpResBlock(nn.Module):
    def __init__(self, in_units, out_units, upsample=True, batch_norm=True):
        super().__init__()
        self.upsample = upsample
        self.batch_norm = batch_norm
        self.block = ResBlock(
            in_units, out_units, 3, dilations=[1, 2, 1], expansion=1)
        if self.upsample:
            self.conv3 = nn.ConvTranspose2d(
                out_units, out_units // 2, kernel_size=2, stride=2,
                padding=0, output_padding=0)
            if self.batch_norm:
                self.bn3 = nn.BatchNorm2d(out_units // 2)


    def forward(self, x):
        x = self.block(x)
        if self.upsample:
            x = F.relu(self.conv3(x), inplace=True)
            if self.batch_norm:
                x = self.bn3(x)
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
        self.dropout = nn.Dropout(p=0.0)
        if self.upsample:
            self.conv3 = nn.ConvTranspose2d(
                out_units, out_units // 2, kernel_size=2, stride=2,
                padding=0, output_padding=0)
            if self.batch_norm:
                self.bn3 = nn.BatchNorm2d(out_units // 2)


    def forward(self, x):
        x = F.relu(self.conv1(x), inplace=True)
        if self.batch_norm:
            x = self.bn1(x)
        x = F.relu(self.conv2(x), inplace=True)
        if self.batch_norm:
            x = self.bn2(x)
        if self.training:
            x = self.dropout(x)
        if self.upsample:
            x = F.relu(self.conv3(x), inplace=True)
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
        input_height, input_width = x.shape[2:]
        down_outputs = []
        for layer in self.down_layers:
            x = layer(x)
            down_outputs.append(x)
            x = F.max_pool2d(x, kernel_size=2)
        x = F.relu(self.interface1(x), inplace=True)
        x = F.relu(self.interface2(x), inplace=True)
        x = F.relu(self.interface_up(x), inplace=True)
        for i, layer in enumerate(self.up_layers):
            skip = down_outputs[-(i+1)]
            x = _maybe_pad(x, skip.shape[2:])
            skip = _maybe_pad(skip, x.shape[2:])
            x = torch.cat([x, skip], dim=1)
            x = layer(x)
        x = F.interpolate(
            x, (input_height, input_width), mode='bilinear', align_corners=True)
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
            UpLayer(skip_channels[-1] + self.bottleneck_channels // 2, up_channels[0]),
            UpLayer(skip_channels[-2] + up_channels[0] // 2, up_channels[1]),
            UpLayer(skip_channels[-3] + up_channels[1] // 2, up_channels[2]),
            UpLayer(skip_channels[-4] + up_channels[2] // 2, up_channels[3])
        ])
        self.conv_classes = nn.Conv2d(up_channels[3] // 2, n_classes, kernel_size=1)

    def _resnet_forward(self, x):
        skips = []
        x = self.network.conv1(x)
        skips.append(x)
        x = self.network.bn1(x)
        x = self.network.relu(x)
        x = self.network.maxpool(x)
        skips.append(x)
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
        for i in range(0, len(self.up_layers)):
            skip = _maybe_pad(skips[-i - 1], x.shape[2:])
            x = torch.cat([x, skip], dim=1)
            x = self.up_layers[i](x)
        x = _maybe_pad(x, input_tensor.shape[2:])
        x = F.interpolate(
            x, size=input_tensor.shape[2:], mode='bilinear', align_corners=True)
        scores = self.conv_classes(x)
        return scores


def unet_resnet(
        resnet_model_fn, in_channels, n_classes, pretrained=True,
        up_channels=[512, 256, 128, 64], **kwargs):
    net = resnet_model_fn(
        in_channels=in_channels, output_stride=32, pretrained=pretrained,
        **kwargs)
    return UNetResNetV1(
        n_classes=n_classes, network=net,
        skip_channels=[64, 64, 512, 1024],
        interface_channels=2048,
        up_channels=up_channels
        )


def unet_resnet101(*args, **kwargs):
    return unet_resnet(resnet101, *args, **kwargs)


def unet_resnet152(*args, **kwargs):
    return unet_resnet(resnet152, *args, **kwargs)

def unet_big_resnet101(*args, **kwargs):
    return unet_resnet(
        resnet101, *args, up_channels=[1024, 256, 128, 64], **kwargs)

