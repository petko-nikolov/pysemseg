from torch.utils.data import Dataset


class DatasetTransformer(Dataset):
    def __init__(
            self, dataset, transform, mode='train'):
        self.dataset = dataset
        self.transform = transform
        self.mode = mode

    def __getitem__(self, index):
        example_id, image, mask = self.dataset[index]
        image, mask = self.transform(image, mask)
        if self.mode == 'test':
            return example_id, image
        return example_id, image, mask

    def __len__(self):
        return len(self.dataset)

    @property
    def labels(self):
        return self.dataset.labels

    @property
    def number_of_classes(self):
        return self.dataset.number_of_classes

    @property
    def color_palette(self):
        return self.dataset.color_palette

    @property
    def ignore_index(self):
        return self.dataset.ignore_index

    @property
    def in_channels(self):
        return self.dataset.in_channels
