from torch.utils.data import Dataset
import torch


class TransformWrapper(Dataset):
    def __init__(
            self, dataset, transform=None, target_transform=None, mode='train'):
        self.dataset = dataset
        self.transform = transform
        self.target_transform = target_transform
        self.mode = mode

    def __getitem__(self, index):
        example_id, input_image, target_mask = self.dataset[index]
        transformed_input = self.transform(input_image)
        if self.mode == 'test':
            return example_id, transformed_input
        transformed_target = self.target_transform(target_mask)
        return example_id, transformed_input, transformed_target

    def __len__(self):
            return self.dataset.__len__()

    @property
    def number_of_classes(self):
        return self.dataset.number_of_classes
