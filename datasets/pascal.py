import os
from torch.utils.data import Dataset


class PascalVOCBase(Dataset):
    def __init__(
            self, root, split='train', transform=None,
            target_transform=None):

        assert split in ['train', 'test', 'val', 'trainval']

        self.root = os.path.expanduser(root)
        self.split = split
        self.transform = transform
        self.target_transform = target_transform


class PascalVOCSegmentation(PascalVOCBase):
    def __init__(self, root, mask='class', *args, **kwargs):
        super().__init__(root, *args, **kwargs)

        ground_truth_dirname = {
            'class': 'SegmentationClass',
            'object': 'SegmentationObject'}[mask]
        self.ground_truth_dir = os.path.join(
            self.root, ground_truth_dirname)
        self.image_dir = os.path.join(self.root, 'JPEGImages')
        self.split_filepath = os.path.join(
            self.root, 'ImageSets/Segmentation', self.split + '.txt')
        self.image_data = self._parse_image_paths(
            self.image_dir, self.ground_truth_dir, self.split_filepath)

    @classmethod
    def _parse_image_paths(cls, image_dir, ground_truth_dir, split_filepath):

        with open(split_filepath) as split_file:
            image_ids = [l.strip() for l in split_file]

        image_data = []

        for image_id in image_ids:
            img_path = os.path.join(image_dir, image_id + '.jpg')
            mask_path = os.path.join(ground_truth_dir, image_id + '.png')
            if os.path.exists(img_path) and os.path.exists(mask_path):
                image_data.append({
                    'id': image_id,
                    'image_filepath': img_path,
                    'gt_filepath': mask_path
                })
        return image_data

    def __getitem__(self, index):
        item = self.image_data[index]
        return item['image_filepath'], item['gt_filepath']

    def __len__(self):
        return len(self.image_data)
