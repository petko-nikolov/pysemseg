import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class VGGFCN32(nn.Module):
    def __init__(self, in_channels, n_classes):
        super().__init__()
        assert in_channels == 3
        self.n_classes = n_classes
        self.vgg16 = models.vgg16(pretrained=True)
        self.classifier = nn.Sequential(
            nn.Conv2d(512, 4096, 7),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Conv2d(4096, 4096, 1),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Conv2d(4096, n_classes, 1),
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

    def forward(self, x):
        input_height, input_width = x.shape[2], x.shape[3]
        x = self.vgg16.features(x)
        x = self.classifier(x)
        x = F.interpolate(x, size=(input_height, input_width), mode='bilinear')
        return x


class VGGFCN16(nn.Module):
    def __init__(self, in_channels, n_classes):
        super().__init__()
        assert in_channels == 3
        self.n_classes = n_classes
        self.vgg16 = models.vgg16(pretrained=True)
        self.classifier = nn.Sequential(
            nn.Conv2d(512, 4096, kernel_size=7, padding=3),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Conv2d(4096, 4096, 1),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Conv2d(4096, n_classes, 1),
        )
        self.score4 = nn.Conv2d(512, n_classes, kernel_size=1)
        self.upscale5 = nn.ConvTranspose2d(
            n_classes, n_classes, kernel_size=2, stride=2)

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

    def forward(self, x):
        input_height, input_width = x.shape[2], x.shape[3]
        pool4 = self.vgg16.features[:-7](x)
        pool5 = self.vgg16.features[-7:](pool4)

        pool5_upscaled = self.upscale5(self.classifier(pool5))
        pool4 = self.score4(pool4)
        x = pool4 + pool5_upscaled
        x = F.interpolate(x, size=(input_height, input_width), mode='bilinear')
        return x


class VGGFCN8(nn.Module):
    def __init__(self, in_channels, n_classes):
        super().__init__()
        assert in_channels == 3
        self.n_classes = n_classes
        self.vgg16 = models.vgg16(pretrained=True)
        self.classifier = nn.Sequential(
            nn.Conv2d(512, 4096, kernel_size=7, padding=3),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Conv2d(4096, 4096, 1),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Conv2d(4096, n_classes, 1),
        )
        self.upscale4 = nn.ConvTranspose2d(
            n_classes, n_classes, kernel_size=2, stride=2)
        self.score4 = nn.Conv2d(
            512, n_classes, kernel_size=1, stride=1)
        self.score3 = nn.Conv2d(
            256, n_classes, kernel_size=1, stride=1)
        self.upscale5 = nn.ConvTranspose2d(
            n_classes, n_classes, kernel_size=2, stride=2)

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

    def forward(self, x):
        input_height, input_width = x.shape[2], x.shape[3]
        pool3 = self.vgg16.features[:-14](x)
        pool4 = self.vgg16.features[-14:-7](pool3)
        pool5 = self.vgg16.features[-7:](pool4)

        pool5_upscaled = self.upscale5(self.classifier(pool5))
        pool4_scores = self.score4(pool4)
        pool4_fused = pool4_scores + pool5_upscaled
        pool4_upscaled = self.upscale4(pool4_fused)
        x = self.score3(pool3) + pool4_upscaled
        x = F.interpolate(x, size=(input_height, input_width), mode='bilinear')
        return x
