import os
from abc import ABCMeta, abstractmethod
from torch.utils.data import Dataset


PASCAL_CLASSES = [
    'background',
    'aeroplane',
    'bicycle',
    'bird',
    'boat',
    'bottle',
    'bus',
    'car',
    'cat',
    'chair',
    'cow',
    'diningtable',
    'dog',
    'horse',
    'motorbike',
    'person',
    'potted-plant',
    'sheep',
    'sofa',
    'train',
    'tv/monitor'
]


def _read_image_ids(split_filepath):
    with open(split_filepath, 'r') as split_file:
        return [s.strip() for s in split_file]


def _parse_image_paths(image_dir, ground_truth_dir, image_ids):
    image_data = []

    for image_id in image_ids:
        img_path = os.path.join(image_dir, image_id + '.jpg')
        mask_path = os.path.join(ground_truth_dir, image_id + '.png')
        if os.path.exists(img_path):
            image_data.append({
                'id': image_id,
                'image_filepath': img_path,
                'gt_filepath':  (
                    mask_path if os.path.exists(mask_path) else None)
            })
    return image_data


class PascalVOCBase(Dataset, metaclass=ABCMeta):
    def __init__(
            self, root, split='train', transform=None,
            target_transform=None):

        assert split in ['train', 'test', 'val']

        self.root = os.path.expanduser(root)
        self.split = split
        self.transform = transform
        self.target_transform = target_transform

    @abstractmethod
    def __getitem__(self, index):
        raise NotImplementedError()

    @abstractmethod
    def __len__(self):
        raise NotImplementedError()

    @property
    def number_of_classes(self):
        return 21


class PascalVOCSegmentation(PascalVOCBase):
    def __init__(self, root, *args, **kwargs):
        super().__init__(root, *args, **kwargs)

        benchmark_train_ids = set(
            _read_image_ids(
                os.path.join(root, 'benchmark_RELEASE/dataset/train.txt')
            )
        )

        benchmark_val_ids = set(
            _read_image_ids(
                os.path.join(root, 'benchmark_RELEASE/dataset/val.txt')
            )
        )

        voc2012_train_ids = set(
            _read_image_ids(
                os.path.join(
                    root,
                    'VOCdevkit/VOC2012/ImageSets/Segmentation/train.txt')
            )
        )

        voc2012_val_ids = set(
            _read_image_ids(
                os.path.join(
                    root,
                    'VOCdevkit/VOC2012/ImageSets/Segmentation/val.txt')
            )
        )

        self.train_image_data = _parse_image_paths(
            os.path.join(root, 'VOCdevkit/VOC2012/JPEGImages'),
            os.path.join(root, 'VOCdevkit/VOC2012/SegmentationClassLabels'),
            voc2012_train_ids
        )

        self.train_image_data.extend(_parse_image_paths(
            os.path.join(root, 'benchmark_RELEASE/dataset/img'),
            os.path.join(root, 'benchmark_RELEASE/dataset/cls_labels'),
            benchmark_train_ids | benchmark_val_ids
        ))

        self.train_ids = (
            voc2012_train_ids | benchmark_val_ids | benchmark_train_ids
        )

        self.val_ids = voc2012_val_ids - self.train_ids

        self.val_image_data = _parse_image_paths(
            os.path.join(root, 'VOCdevkit/VOC2012/JPEGImages'),
            os.path.join(root, 'VOCdevkit/VOC2012/SegmentationClassLabels'),
            self.val_ids
        )

        self.image_data = {
            'train': self.train_image_data,
            'val': self.val_image_data
        }[self.split]

    def __getitem__(self, index):
        item = self.image_data[index]
        return item['id'], item['image_filepath'], item['gt_filepath']

    def __len__(self):
        return len(self.image_data)
