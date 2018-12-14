import torch
import torch.nn as nn
import torch.nn.functional as F
from .resnet import resnet50, resnet101, resnet152

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
            nn.AdaptiveAvgPool2d((1, 1)),
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
        x5 = F.interpolate(
            x5, size=x1.size()[2:], mode='bilinear', align_corners=True
        )
        x = torch.cat([x1, x2, x3, x4, x5], dim=1)
        x = self.relu(self.bn(x))
        x = self.final_conv(x)
        return x


class Decoder(nn.Module):
    def __init__(self, low_level_channels, encoder_channels,
                 low_level_reduced_channels=48):
        super().__init__()
        self.reduce_low_level = nn.Conv2d(
            low_level_channels, low_level_reduced_channels, kernel_size=1)
        self.fuse = nn.Sequential(
            nn.Conv2d(
                encoder_channels + low_level_reduced_channels, 256,
                kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(256),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )

    def forward(self, x, low_level_features):
        low_level_features = self.reduce_low_level(low_level_features)
        x = F.interpolate(
            x, size=low_level_features.shape[2:], mode='bilinear',
            align_corners=True
        )
        x = torch.cat([x, low_level_features], dim=1)
        x = self.fuse(x)
        return x


class Deeplab(nn.Module):
    def __init__(
            self, in_channels, n_classes, backbone_cls, decoder=None,
            score_channels=256, aspp_rates=[6, 12, 18], pretrained=True,
            output_stride=16, multi_grid=[1, 2, 4], finetune_bn=True):
        super().__init__()
        self.backbone = backbone_cls(
            pretrained=pretrained,
            in_channels=in_channels,
            output_stride=output_stride,
            multi_grid=multi_grid
        )
        rate = 2 if output_stride == 8 else 1
        self.aspp = ASPPModule(2048, 256, [r * rate for r in aspp_rates])
        self.finetune_bn = finetune_bn
        self.decoder = decoder
        self.score = nn.Sequential(
            nn.Conv2d(score_channels, score_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(score_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(score_channels, n_classes, kernel_size=1)
        )

    def forward(self, x):
        input_size = x.shape[2:]
        x, block_outputs = self.backbone(x)
        x = self.aspp(x)
        if self.decoder:
            x = self.decoder(x, block_outputs[0])
        x = F.interpolate(
            x, size=input_size, mode='bilinear', align_corners=True
        )
        x = self.score(x)
        return x

    def train(self, mode=True):
        super().train(mode)
        if not self.finetune_bn:
            for module in self.modules():
                if isinstance(module, nn.BatchNorm2d):
                    module.train(False)


def deeplabv3_resnet50(in_channels, n_classes, **kwargs):
    return Deeplab(in_channels, n_classes, resnet50, **kwargs)


def deeplabv3_resnet101(in_channels, n_classes, **kwargs):
    return Deeplab(in_channels, n_classes, resnet101, **kwargs)


def deeplabv3_resnet152(in_channels, n_classes, **kwargs):
    return Deeplab(in_channels, n_classes, resnet152, **kwargs)


def deeplabv3plus_resnet101(in_channels, n_classes, **kwargs):
    return Deeplab(
        in_channels, n_classes, resnet101,
        decoder=Decoder(256, 256, 48), **kwargs
    )


def deeplabv3plus_resnet50(in_channels, n_classes, **kwargs):
    return Deeplab(
        in_channels, n_classes, resnet50,
        decoder=Decoder(256, 256, 48), **kwargs
    )
