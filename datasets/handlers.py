"""
A dataset handler accepts a data directory and mode and returns a
torch.utils.data.Dataset object
The accepted values for mode are train, val and test
"""


import cv2
from transforms import (
    CV2ImageLoader, Grayscale, Resize, ToCategoryTensor, ToTensor, Binarize,
    RandomContrast, RandomBrightness, RandomHueSaturation, ToFloatImage)
from datasets.pascal_voc import PascalVOCSegmentation
from datasets.transformer import TransformWrapper
from torchvision.transforms import Compose, Normalize


def pascal_voc(data_dir, mode):
    return TransformWrapper(
        PascalVOCSegmentation(data_dir, split=mode),
        transform=Compose([
            CV2ImageLoader(),
            ToFloatImage(),
            Resize((512, 512)),
            ToTensor(),
             Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )]),
        target_transform=Compose([
            CV2ImageLoader(grayscale=True),
            Resize((512, 512), interpolation=cv2.INTER_NEAREST),
            ToCategoryTensor()]),
        mode=mode)
