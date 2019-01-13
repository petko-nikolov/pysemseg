import numpy as np
import torch
from pysemseg.utils import tensor_to_numpy

def _apply(fcn, acc, counter):
    result = {}
    for key, value in acc.items():
        if isinstance(value, dict):
            result[key] = _apply(fcn, value, counter.get(key, {}))
        else:
            result[key] = fcn(acc[key], counter.get(key, 0.0))
    return result


class _Accumulator():
    def __init__(self):
        self.numerator = {}
        self.denominator = {}

    def mean(self):
        return _apply(
            lambda n, d: n / d if d > 0 else 0.0,
            self.numerator, self.denominator)

    def update(self, data):
        self.numerator = _apply(
            lambda a, d: a[0] + d, data, self.numerator)
        self.denominator = _apply(
            lambda a, d: a[1] + d,
            data, self.denominator)


class SegmentationMetrics:
    def __init__(self, num_classes, labels=None, ignore_index=-1):
        labels = labels or list(range(num_classes))
        self.labels = dict(enumerate(labels))
        self.accumulator = _Accumulator()
        self.num_classes = num_classes
        self.cm = np.zeros(
            (self.num_classes, self.num_classes), dtype=np.int32)
        self.ignore_index = ignore_index


    def metrics(self):
        metrics = {}
        metrics['accuracy'] = self._accuracy()
        metrics['class'] = {}
        metrics['class']['iou'] = dict(zip(self.labels, self._iou()))
        metrics['mIOU'] = np.mean(list(metrics['class']['iou'].values()))
        return metrics

    def _accuracy(self):
        return self.cm.diagonal().sum() / self.cm.sum()

    def _iou(self):
        colsum = self.cm.sum(axis=0)
        rowsum = self.cm.sum(axis=1)
        diag = self.cm.diagonal()
        return diag / (colsum + rowsum - diag)

    def _confusion_matrix(self, outputs, targets):
        outputs = outputs.reshape(-1,)
        targets = targets.reshape(-1,)
        mask = targets != self.ignore_index
        comb = self.num_classes * outputs[mask] + targets[mask]
        comb = np.bincount(comb, minlength=self.num_classes ** 2)
        return comb.reshape(self.num_classes, self.num_classes)


    def add(self, outputs, targets, loss):
        self.accumulator.update({'loss': (loss, 1)})
        self.cm += self._confusion_matrix(outputs, targets)


class TorchSegmentationMetrics:
    def __init__(self, num_classes, labels=None, ignore_index=-1, device=None):
        labels = labels or list(range(num_classes))
        self.labels = dict(enumerate(labels))
        self.accumulator = _Accumulator()
        self.num_classes = num_classes
        self.cm = torch.zeros(
            (self.num_classes, self.num_classes), dtype=torch.long,
            requires_grad=False, device=device)
        self.ignore_index = ignore_index


    def metrics(self):
        metrics = {}
        metrics['accuracy'] = self._accuracy()
        metrics['class'] = {
            k: {'iou': v}
            for k, v in zip(self.labels, self._iou())
        }
        metrics['mIOU'] = np.mean([c['iou'] for c in metrics['class'].values()])
        metrics.update(self.accumulator.mean())
        metrics = _apply(
            lambda x, y: float(x),
            metrics,
            {}
        )
        return metrics

    def _accuracy(self):
        accuracy = self.cm.diagonal().sum().float() / self.cm.sum().float()
        return tensor_to_numpy(accuracy)

    def _iou(self):
        colsum = self.cm.sum(dim=0)
        rowsum = self.cm.sum(dim=1)
        diag = self.cm.diagonal()
        iou = diag.float() / (colsum + rowsum - diag).float()
        return tensor_to_numpy(iou)

    def _confusion_matrix(self, outputs, targets):
        outputs = outputs.view(-1).contiguous()
        targets = targets.view(-1).contiguous()
        mask = (targets != self.ignore_index)
        comb = self.num_classes * outputs[mask] + targets[mask]
        comb = torch.bincount(comb, minlength=self.num_classes ** 2)
        return comb.reshape(self.num_classes, self.num_classes).long()


    def add(self, outputs, targets, loss):
        self.accumulator.update({'loss': (loss, 1)})
        self.cm += self._confusion_matrix(outputs, targets)


def compute_example_segmentation_metrics(
        n_classes, outputs, targets, loss, ignore_index=-1):
    metrics = TorchSegmentationMetrics(n_classes, ignore_index)
    metrics.add(outputs, targets, loss)
    return metrics.metrics()
