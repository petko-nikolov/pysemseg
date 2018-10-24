import shutil
import sys
import os
import numpy as np
import torch


def prompt_delete_dir(directory):
    if os.path.exists(directory):
        answer = input(
            "{} exists. Do you want to delete it?[y/n]".format(directory))
        if answer == 'y':
            shutil.rmtree(directory)
        elif answer != 'n':
            sys.exit(1)


def restore(checkpoint_path, model, optimizer=None, restore_cpu=False):
    checkpoint = torch.load(
        checkpoint_path,
        map_location=lambda storage, location: storage if restore_cpu else None)
    model.load_state_dict(checkpoint['state'])
    if optimizer:
        optimizer.load_state_dict(checkpoint['optimizer'])
    return checkpoint['epoch']


def tensor_to_numpy(tensor):
    if tensor.is_cuda:
        tensor = tensor.cpu()
    return tensor.numpy()


def import_class_module(name):
    components = name.split('.')
    mod = __import__(components[0])
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod


def flatten_dict(dict_obj):
    result = {}
    for key, value in dict_obj.items():
        if isinstance(value, dict):
            flattened = flatten_dict(value)
            result.update({
                "{}/{}".format(key, kk): fv
                for kk, fv in flattened.items()})
        else:
            result[key] = value
    return result


def _get_palette_map(n_classes):
    def bitget(byteval, idx):
        return (byteval & (1 << idx)) != 0
    color_to_label = {}
    for k in range(0, n_classes):
        red = green = blue = 0
        cls = k
        for j in range(8):
            red = red | (bitget(cls, 0) << 7 - j)
            green = green | (bitget(cls, 1) << 7 - j)
            blue = blue | (bitget(cls, 2) << 7 - j)
            cls = cls >> 3
        color_to_label[(red, green, blue)] = k
    return color_to_label


class ColorPalette256:
    def __init__(self, n_classes):
        self.color_to_label = _get_palette_map(n_classes)
        self.label_to_color = {v: k for k, v in self.color_to_label.items()}

    def encode_color(self, label_mask):
        """
        Encodes a label mask with its RGB color representation
        Arguments:
            label_mask: A numpy array with dimensions either
            (height, width), (height, width, 1), (batch_size, height, width) or
            (batch_size, height, width, 1)

        Returns: Color encoded representation
        """
        input_shape = label_mask.shape
        if len(input_shape) == 2:
            label_mask = np.reshape(label_mask, [1, *input_shape, 1])
            output_shape = input_shape + (3,)
        elif len(input_shape) == 3 and input_shape[-1] == 1:
            label_mask = np.expand_dims(label_mask, axis=0)
            output_shape = input_shape[:2] + (3,)
        elif len(input_shape) == 3:
            label_mask = np.expand_dims(label_mask, axis=-1)
            output_shape = input_shape + (3,)
        elif len(input_shape) == 4:
            output_shape = input_shape[:3] + (3,)

        palette_mask = np.zeros(label_mask.shape[:3] + (3,), dtype=np.uint8)
        for i, mask in self.label_to_color.items():
            palette_mask[np.where(np.all(label_mask == i, axis=-1))[:3]] = mask

        palette_mask = palette_mask.reshape(output_shape)

        return palette_mask


    def decode_color(self, palette_mask):
        label_mask = np.zeros(palette_mask.shape[:2], dtype=np.uint8)
        for mask, i in self.color_to_label.items():
            label_mask[np.where(np.all(palette_mask == mask, axis=-1))[:2]] = i
        return label_mask
