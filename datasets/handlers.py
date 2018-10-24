"""
A dataset handler accepts a data directory and mode and returns a
torch.utils.data.Dataset object
The accepted values for mode are train, val and test
"""


from datasets.pascal_voc import PascalVOCSegmentation, PascalVOCTransform
from datasets.transformer import DatasetTransformer


def pascal_voc(data_dir, mode):
    return DatasetTransformer(
        PascalVOCSegmentation(data_dir, split=mode),
        transform=PascalVOCTransform(mode),
        mode=mode)
