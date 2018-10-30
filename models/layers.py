import torch.nn as nn
import torch.nn.functional as F

class PaddingSameConv2d(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x):
        x = F.pad(
            x,
            [self.kernel_size[0]//2] * 2 +
            [self.kernel_size[1]//2] * 2)
        return super().forward(x)

