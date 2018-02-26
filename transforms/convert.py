import numpy as np
import cv2
import torch


class Grayscale:
    def __call__(self, image):
        return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)


class ToCategoryTensor:
    def __init__(self, remap=None):
        self.remap = remap

    def __call__(self, image):
        if self.remap:
            for k, v in self.remap.items():
                image[image == k] = v
        return torch.LongTensor(image)


class ToTensor:
    def __init__(self, normalize=True):
        self.normalize = normalize

    def __call__(self, image):
        if len(image.shape) == 2:
            image = np.expand_dims(image, axis=-1)
        if self.normalize:
            image = image / 255.0
        image = np.transpose(image, [2, 0, 1])
        return torch.FloatTensor(image)
