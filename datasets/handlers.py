from .pascal import PascalVOCSegmentation
from .transformer import TransformWrapper
from torchvision.transforms import Compose, ToTensor
from ..transforms import PILImageLoader

"""
A dataset handler accepts a data directory and mode and returns a
torch.utils.data.Dataset object
The accepted values for mode are train, val and test
"""


def _pascal_voc(data_dir, mode, mask):
    return TransformWrapper(
        PascalVOCSegmentation(data_dir, split=mode, mask=mask),
        transform=Compose([
            PILImageLoader(),
            ToTensor()]),
        target_transform=Compose([
            PILImageLoader(),
            ToTensor()]))


def pascal_voc_class(data_dir, mode):
    return _pascal_voc(data_dir, mode, 'class')


def pascal_voc_object(data_dir, mode):
    return _pascal_voc(data_dir, mode, 'object')
