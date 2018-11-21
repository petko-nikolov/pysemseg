from .base import SegmentationDataset
from .pascal_voc.pascal import PascalVOCSegmentation, PascalVOCTransform
from .camvid import CamVid, CamVidTransform
from .transformer import DatasetTransformer


def create_dataset(data_dir, dataset_cls, transformer_cls, mode):
    dataset = dataset_cls(data_dir, mode)
    transformer = transformer_cls(mode)
    return DatasetTransformer(dataset, transformer, mode)
