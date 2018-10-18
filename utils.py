import shutil
import sys
import os
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
