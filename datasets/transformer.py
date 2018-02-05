from torch.utils.data import Dataset


class TransformWrapper(Dataset):
    def __init__(
            self, dataset, transform=None, target_transform=None):
        self.dataset = dataset
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        input_image, target_mask = self.dataset[index]
        transformed_input = self.transform(input_image)
        transformed_target = self.target_transform(target_mask)
        return transformed_input, transformed_target

    def __len__(self):
            return self.dataset.__len__()
