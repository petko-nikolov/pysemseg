import os
from torchvision.transforms import Normalize
from datasets.base import SegmentationDataset
import cv2
import transforms


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


class PascalVOCSegmentation(SegmentationDataset):
    def __init__(self, root, split='train'):
        super().__init__()

        assert split in ['train', 'test', 'val']
        self.root = os.path.expanduser(root)
        self.split = split

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

    @property
    def number_of_classes(self):
        return 21

    @property
    def labels(self):
        return PASCAL_CLASSES

    @property
    def ignore_index(self):
        return 255

    def __getitem__(self, index):
        item = self.image_data[index]
        return item['id'], item['image_filepath'], item['gt_filepath']

    def __len__(self):
        return len(self.image_data)


class PascalVOCTransform:
    def __init__(self, mode, ignore_index=-1):
        self.mode = mode
        self.image_loader = transforms.Compose([
            transforms.CV2ImageLoader(),
            transforms.PadTo((224, 224)),
            transforms.ToFloatImage()
        ])

        self.target_loader = transforms.Compose([
            transforms.CV2ImageLoader(grayscale=True),
            transforms.PadTo((224, 224), pad_value=ignore_index),
        ])

        self.crop = transforms.RandomCrop((224, 224))

        self.image_augmentations = transforms.Compose([
            transforms.RandomHueSaturation(
                hue_delta=0.05, saturation_scale_range=(0.7, 1.3)),
            transforms.RandomContrast(0.5, 1.5),
            transforms.RandomBrightness(-32.0 / 255, 32. / 255)
        ])

        self.joint_augmentations = transforms.Compose(
            [transforms.RandomHorizontalFlip()]
        )

        self.tensor_transforms = transforms.Compose([
            transforms.ToTensor(),
            Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225])
        ])

    def __call__(self, image, target):
        image = self.image_loader(image)
        target = self.target_loader(target)
        image, target = self.crop(image, target)
        if self.mode == 'train':
            image = self.image_augmentations(image)
            image, target = self.joint_augmentations(image, target)
        image = self.tensor_transforms(image)
        target = transforms.ToCategoryTensor()(target)
        return image, target
