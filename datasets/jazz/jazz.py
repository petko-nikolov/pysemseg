import os
import glob
from torch.utils.data import Dataset


def parse_image_data(root_dir):
    images_data = []
    for image_path in glob.glob(
            os.path.join(root_dir, '*/*.tif')):
        if any(s in image_path for s in ['mask', 'ref']):
            continue
        image_id, _ = os.path.basename(image_path).split('.')
        image_dir = os.path.dirname(image_path)
        image_data = {
            'id': image_id,
            'image_filepath': image_path,
            'ref_image_filepath': os.path.join(
                image_dir, image_id + '_ref.tif'),
            'mask_image_filepath': os.path.join(
                image_dir, image_id + '_mask.tif')
        }
        images_data.append(image_data)
    return images_data


class JazzSegmentationDataset(Dataset):
    def __init__(self, root, mode):
        self.root = root
        self.ground_truth_dir = os.path.join(
            self.root, mode)
        self.image_data = parse_image_data(self.ground_truth_dir)

    def __getitem__(self, index):
        item = self.image_data[index]
        return item['id'], item['image_filepath'], item['mask_image_filepath']

    def __len__(self):
        return len(self.image_data)

    @property
    def number_of_classes(self):
        return 2
