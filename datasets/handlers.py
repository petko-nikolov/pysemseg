from datasets.pascal import PascalVOCSegmentation
from datasets.jazz import JazzSegmentationDataset
from datasets.transformer import TransformWrapper
from torchvision.transforms import Compose, Normalize
from transforms import (
    CV2ImageLoader, Grayscale, Resize, ToCategoryTensor, ToTensor, Binarize,
    RandomContrast, RandomBrightness, RandomHueSaturation, ToFloatImage)
import cv2

"""
A dataset handler accepts a data directory and mode and returns a
torch.utils.data.Dataset object
The accepted values for mode are train, val and test
"""



def _pascal_voc(data_dir, mode, mask):
    return TransformWrapper(
        PascalVOCSegmentation(data_dir, split=mode, mask=mask),
        transform=Compose([
            CV2ImageLoader(),
            Resize((256, 256)),
            ToTensor()]),
        target_transform=Compose([
            CV2ImageLoader(),
            Grayscale(),
            Resize((256, 256), interpolation=cv2.INTER_NEAREST),
            ToCategoryTensor()]),
        mode=mode)


def pascal_voc_class(data_dir, mode):
    return _pascal_voc(data_dir, mode, 'class')


def pascal_voc_object(data_dir, mode):
    return _pascal_voc(data_dir, mode, 'object')


def jazz(data_dir, mode):
    return TransformWrapper(
        JazzSegmentationDataset(data_dir, mode=mode),
        transform=Compose([
            CV2ImageLoader(),
            Resize((256, 256)),
            ToFloatImage(),
            RandomContrast(0.8, 1.2),
            RandomBrightness(-0.3, 0.3),
            RandomHueSaturation((-0.05, 0.05), (-0.15, 0.15)),
            ToTensor(),
            Normalize(-1.0, 1.0)]),
        target_transform=Compose([
            CV2ImageLoader(grayscale=True),
            Resize((256, 256)),
            Binarize(127),
            ToCategoryTensor(remap={255: 1})]),
        mode=mode)
