import torch
import shutil
import sys
import os

def prompt_delete_dir(directory):
    if os.path.exists(directory):
        answer = input(
            "{} exists. Do you want to delete it?[y/n]".format(directory))
        if answer == 'y':
            shutil.rmtree(directory)
        elif answer != 'n':
            sys.exit(1)


def restore(checkpoint_path, model, optimizer=None):
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['state'])
    if optimizer:
        optimizer.load_state_dict(checkpoint['optimizer'])
    return checkpoint['epoch']


