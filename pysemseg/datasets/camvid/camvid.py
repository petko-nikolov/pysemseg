import os
import glob
from torchvision.transforms import Normalize
import cv2

from pysemseg import transforms
from pysemseg.datasets.base import SegmentationDataset
from pysemseg.utils import ColorPalette


CAMVID_CLASSES = [
    "Sky",
    "Building",
    "Column_pole",
    "Road",
    "Sidewalk",
    "Tree",
    "SignSymbol",
    "Fence",
    "Car",
    "Pedestrian",
    "Bicyclist"
]

CAMVID_COLORS = [
    (128, 128, 128),
    (128, 0, 0),
    (192, 192, 128),
    (128, 64, 128),
    (0, 0, 192),
    (128, 128, 0),
    (192, 128, 128),
    (64, 64, 128),
    (64, 0, 128),
    (64, 64, 0),
    (0, 128, 192),
    (0, 0, 0)
]


def _parse_image_paths(images_dir, annotations_dir):
    image_data = []
    for image_filepath in glob.glob(images_dir + '/*.png'):
        image_filename = os.path.basename(image_filepath)
        annotation_filepath = os.path.join(annotations_dir, image_filename)
        assert os.path.exists(annotation_filepath)
        image_data.append({
            'id': image_filename,
            'image_filepath': image_filepath,
            'gt_filepath':  (
                annotation_filepath
                if os.path.exists(annotation_filepath) else None)
        })
    return image_data


class CamVid(SegmentationDataset):
    def __init__(self, root_dir, split):
        super().__init__()
        assert split in ['train', 'val', 'test']
        self.color_palette_ = ColorPalette(CAMVID_COLORS)
        self.image_loader = CV2ImageLoader()
        self.target_laoder = CV2ImageLoader(grayscale=True)
        self.root_dir = root_dir
        self.split = split
        self.image_data = _parse_image_paths(
            os.path.join(self.root_dir, split),
            os.path.join(self.root_dir, split + 'annot')
        )

    @property
    def number_of_classes(self):
        return 12

    @property
    def labels(self):
        return CAMVID_CLASSES

    @property
    def ignore_index(self):
        return 11

    def __getitem__(self, index):
        item = self.image_data[index]
        return (
            item['id'],
            self.image_loader(item['image_filepath']),
            self.target_loader(item['gt_filepath'])
        )

    def __len__(self):
        return len(self.image_data)



class CamVidTransform:
    def __init__(self, mode):
        self.mode = mode
        self.image_loader = transforms.Compose([
            transforms.ToFloatImage()
        ])

        self.image_augmentations = transforms.Compose([
            transforms.RandomHueSaturation(
                hue_delta=0.05, saturation_scale_range=(0.7, 1.3)),
            transforms.RandomContrast(0.5, 1.5),
            transforms.RandomBrightness(-32.0 / 255, 32. / 255)
        ])

        self.joint_augmentations = transforms.Compose([
             transforms.RandomCropFixedSize((224, 224)),
             transforms.RandomHorizontalFlip()
        ])

        self.tensor_transforms = transforms.Compose([
            transforms.ToTensor(),
            Normalize(
                mean=[0.39068785, 0.40521392, 0.41434407],
                std=[0.29652068, 0.30514979, 0.30080369])
        ])

    def __call__(self, image, target):
        image = self.image_loader(image)
        if self.mode == 'train':
            image, target = self.joint_augmentations(image, target)
            image = self.image_augmentations(image)
        image = self.tensor_transforms(image)
        target = transforms.ToCategoryTensor()(target)
        return image, target
