from torch.utils.data import Dataset
from abc import ABCMeta, abstractmethod
from pysemseg.utils import ColorPalette


class SegmentationDataset(Dataset, metaclass=ABCMeta):
    def __init__(self):
        self.color_palette_ = ColorPalette(256)

    @property
    @abstractmethod
    def number_of_classes(self):
        pass

    @property
    def labels(self):
        return None

    @property
    def color_palette(self):
        return self.color_palette_

    @property
    def ignore_index(self):
        return -1

    @property
    def in_channels(self):
        return 3
