"""
A dataset handler accepts a data directory and mode and returns a
torch.utils.data.Dataset object
The accepted values for mode are train, val and test
"""


from datasets.pascal_voc import PascalVOCSegmentation, PascalVOCTransform
from datasets.camvid import CamVid, CamVidTransform
from datasets.transformer import DatasetTransformer


def pascal_voc(data_dir, mode):
    dataset = PascalVOCSegmentation(data_dir, split=mode)
    return DatasetTransformer(
        dataset,
        transform=PascalVOCTransform(mode, dataset.ignore_index),
        mode=mode)


def camvid(data_dir, mode):
    return DatasetTransformer(
        CamVid(data_dir, mode),
        transform=CamVidTransform(mode),
        mode=mode)
