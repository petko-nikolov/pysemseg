from torch.optim.lr_scheduler import _LRScheduler


class PolyLR(_LRScheduler):
    def __init__(self, optimizer, max_epochs, gamma=0.9, last_epoch=-1):
        self.max_epochs = max_epochs
        self.gamma = gamma
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        return [
            base_lr * (1.0 - self.last_epoch / self.max_epochs) ** self.gamma
            for base_lr in self.base_lrs
        ]
