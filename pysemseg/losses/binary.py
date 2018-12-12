import torch
import torch.nn as nn
import torch.nn.functional as F


def _compute_dice(inputs, targets, ignore_index):
    assert inputs.shape[1] == 2, "Only binary problems supported"
    smooth = 1.
    inputs = F.softmax(inputs, dim=1)
    foreground_probs = inputs[:, 1, :, :]
    targets = targets.float()
    mask = (targets != ignore_index).float()
    intersection = (foreground_probs * targets * mask).sum()
    denom = (foreground_probs ** 2 * mask + targets ** 2 * mask).sum()
    return (2 * intersection + smooth)/ (denom + smooth)


def _compute_jaccard(inputs, targets, ignore_index):
    assert inputs.shape[1] == 2, "Only binary problems supported"
    smooth = 1.
    inputs = F.softmax(inputs, dim=1)
    foreground_probs = inputs[:, 1, :, :]
    targets = targets.float()
    mask = (targets != ignore_index).float()
    intersection = (foreground_probs * targets * mask).sum()
    union = (foreground_probs * mask + targets * mask).sum()
    return (intersection + smooth) / (union - intersection + smooth)


class DiceLoss(nn.Module):
    def __init__(self, ignore_index=-1, rescale=True):
        super().__init__()
        self.ignore_index = ignore_index
        self.rescale = rescale

    def forward(self, inputs, targets):
        loss = 1.0 - _compute_dice(inputs, targets, self.ignore_index)
        if self.rescale:
            num_targets = torch.sum(targets != self.ignore_index).float()
            loss *= num_targets
        return loss


class LogDiceLoss(nn.Module):
    def __init__(self, ignore_index=-1, rescale=True):
        super().__init__()
        self.ignore_index = ignore_index
        self.rescale = rescale

    def forward(self, inputs, targets):
        loss = -torch.log(_compute_dice(inputs, targets, self.ignore_index))
        if self.rescale:
            num_targets = torch.sum(targets != self.ignore_index).float()
            loss *= num_targets
        return loss


class JaccardLoss(nn.Module):
    def __init__(self, ignore_index=-1, rescale=True):
        super().__init__()
        self.ignore_index = ignore_index
        self.rescale = rescale

    def forward(self, inputs, targets):
        loss = 1.0 - _compute_jaccard(inputs, targets, self.ignore_index)
        if self.rescale:
            num_targets = torch.sum(targets != self.ignore_index).float()
            loss *= num_targets
        return loss


class LogJaccardLoss(nn.Module):
    def __init__(self, ignore_index=-1, rescale=True):
        super().__init__()
        self.ignore_index = ignore_index
        self.rescale = rescale

    def forward(self, inputs, targets):
        loss = -torch.log(_compute_jaccard(inputs, targets, self.ignore_index))
        if self.rescale:
            num_targets = torch.sum(targets != self.ignore_index).float()
            loss *= num_targets
        return loss
