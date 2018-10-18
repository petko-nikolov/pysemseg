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
