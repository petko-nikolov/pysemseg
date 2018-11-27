import torch.nn as nn
import torch.nn.functional as F
from abc import ABCMeta
import torchvision.models as models


def _maybe_pad(x, size):
    hpad = size[0] - x.shape[2]
    wpad = size[1] - x.shape[3]
    if hpad + wpad > 0:
        x = F.pad(x, (0, wpad, 0, hpad, 0, 0, 0, 0 ))
    return x


class VGGFCN(nn.Module, metaclass=ABCMeta):
    def __init__(self, in_channels, n_classes):
        super().__init__()
        assert in_channels == 3
        self.n_classes = n_classes
        self.vgg16 = models.vgg16(pretrained=True)
        self.classifier = nn.Sequential(
            nn.Conv2d(512, 4096, kernel_size=7, padding=3),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Conv2d(4096, 4096, kernel_size=1),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Conv2d(4096, n_classes, kernel_size=1),
        )

        self._initialize_weights()

    def _initialize_weights(self):
        self.classifier[0].weight.data = (
            self.vgg16.classifier[0].weight.data.view(
                self.classifier[0].weight.size())
        )
        self.classifier[3].weight.data = (
            self.vgg16.classifier[3].weight.data.view(
                self.classifier[3].weight.size())
        )


class VGGFCN32(VGGFCN):
    def forward(self, x):
        input_height, input_width = x.shape[2], x.shape[3]
        x = self.vgg16.features(x)
        x = self.classifier(x)
        x = F.interpolate(x, size=(input_height, input_width),
                          mode='bilinear', align_corners=True)
        return x


class VGGFCN16(VGGFCN):
    def __init__(self, in_channels, n_classes):
        super().__init__(in_channels, n_classes)
        self.score4 = nn.Conv2d(512, n_classes, kernel_size=1)
        self.upscale5 = nn.ConvTranspose2d(
            n_classes, n_classes, kernel_size=2, stride=2)

    def forward(self, x):
        input_height, input_width = x.shape[2], x.shape[3]
        pool4 = self.vgg16.features[:-7](x)
        pool5 = self.vgg16.features[-7:](pool4)
        pool5_upscaled = self.upscale5(self.classifier(pool5))
        pool4 = self.score4(pool4)
        x = pool4 + pool5_upscaled
        x = F.interpolate(x, size=(input_height, input_width),
                          mode='bilinear', align_corners=True)
        return x


class VGGFCN8(VGGFCN):
    def __init__(self, in_channels, n_classes):
        super().__init__(in_channels, n_classes)
        self.upscale4 = nn.ConvTranspose2d(
            n_classes, n_classes, kernel_size=2, stride=2)
        self.score4 = nn.Conv2d(
            512, n_classes, kernel_size=1, stride=1)
        self.score3 = nn.Conv2d(
            256, n_classes, kernel_size=1, stride=1)
        self.upscale5 = nn.ConvTranspose2d(
            n_classes, n_classes, kernel_size=2, stride=2)

    def forward(self, x):
        input_height, input_width = x.shape[2], x.shape[3]
        pool3 = self.vgg16.features[:-14](x)
        pool4 = self.vgg16.features[-14:-7](pool3)
        pool5 = self.vgg16.features[-7:](pool4)
        pool5_upscaled = self.upscale5(self.classifier(pool5))
        pool5_upscaled = _maybe_pad(pool5_upscaled, pool4.shape[2:])
        pool4_scores = self.score4(pool4)
        pool4_fused = pool4_scores + pool5_upscaled
        pool4_upscaled = self.upscale4(pool4_fused)
        pool4_upscaled = _maybe_pad(pool4_upscaled, pool3.shape[2:])
        x = self.score3(pool3) + pool4_upscaled
        x = F.interpolate(x, size=(input_height, input_width),
                          mode='bilinear', align_corners=True)
        return x
