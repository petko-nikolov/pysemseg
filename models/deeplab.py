import copy
import torch.nn as nn
from torchvision.models import resnet50


class DeepLabv3(nn.Module):
    def __init__(self, in_channels, n_classes, output_scale=16, multi_grid=(6, 12, 18)):
        assert in_channels == 3
        super().__init__()
        self.resnet = resnet50(pretrained=True)
        # self.resnet.layer4[0].conv2.dilation = (2, 2)
        self.resnet.layer4[0].conv2.stride = (1, 1)
        # self.resnet.layer4[0].downsample[0].stride = (1,1)
        # self.resnet.layer4[1].conv2.dilation = (4, 4)
        # self.resnet.layer4[2].conv2.dilation = (8, 8)

    def forward(self, x):
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        x = self.resnet.layer1(x)
        x = self.resnet.layer2(x)
        x = self.resnet.layer3(x)
        x = self.resnet.layer4(x)

        return x
