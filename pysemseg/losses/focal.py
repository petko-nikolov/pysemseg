import torch
import torch.nn as nn
import torch.nn.functional as F

def one_hot_encode(targets, N):
    targets = targets.unsqueeze(1)
    one_hot = torch.zeros(
        (targets.size(0), N, targets.size(2), targets.size(3)),
        device=targets.device
    )
    one_hot = one_hot.scatter_(1, targets.data, 1)
    one_hot = torch.autograd.Variable(one_hot)
    return one_hot


class FocalLoss(nn.Module):
    def __init__(self, gamma=1.0, weights=None, ignore_index=-1):
        super().__init__()
        self.gamma = gamma
        self.weights = weights
        self.ignore_index = ignore_index
        self.ce = nn.CrossEntropyLoss(
            weight=self.weights, ignore_index=ignore_index, reduction='none'
        )

    def forward(self, inputs, targets):
        ce_loss = self.ce(inputs, targets)
        one_hot = one_hot_encode(targets, inputs.size(1))
        inv_probs = 1.0 - F.softmax(inputs, dim=1)
        focal = (inv_probs * one_hot).sum(dim=1) ** self.gamma
        mask = (targets != self.ignore_index).float()
        return torch.sum(focal * ce_loss * mask)
