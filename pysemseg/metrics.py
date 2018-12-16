import numpy as np
import torch

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
        self.labels = dict(enumerate(labels)) if labels else {}
        self.accumulator = _Accumulator()
        self.num_classes = num_classes
        self.ignore_index = ignore_index


    def metrics(self):
        metrics = self.accumulator.mean()
        metrics['mIOU'] = np.mean(
            [c['iou'] for _, c in metrics['class'].items()]
        )
        return metrics

    def _accuracy(self, outputs, targets):
        mask = (targets != self.ignore_index)
        num = np.sum(np.logical_and(outputs == targets, mask))
        denom = np.sum(mask)
        return num, denom

    def _recall(self, outputs, targets):
        matches = (outputs == targets)
        return [
            (np.sum(matches[targets == i]), np.sum(targets == i))
            for i in range(self.num_classes)
        ]

    def _precision(self, outputs, targets):
        mask = (targets != self.ignore_index)
        matches = (outputs == targets)
        return [(np.sum(matches[np.logical_and(outputs == i, mask)]),
                 np.sum(np.logical_and(outputs == i, mask)))
                for i in range(self.num_classes)]

    def _iou(self, outputs, targets):
        mask = targets != self.ignore_index
        matches = (outputs == targets)
        return [
            (np.sum(matches[targets == i]),
             np.sum(
                 np.logical_and(
                     np.logical_or(outputs == i, targets == i),
                     mask)))
            for i in range(self.num_classes)
        ]

    def add(self, outputs, targets, loss):
        metrics = {'loss': (loss, 1)}
        metrics['accuracy'] = self._accuracy(outputs, targets)
        recall = self._recall(outputs, targets)
        precision = self._precision(outputs, targets)
        iou = self._iou(outputs, targets)
        metrics['class'] = {
            self.labels.get(i, i): {
                'iou': iou[i],
            }
            for i in range(self.num_classes)}
        self.accumulator.update(metrics)


class TorchSegmentationMetrics:
    def __init__(self, num_classes, labels=None, ignore_index=-1):
        self.labels = dict(enumerate(labels)) if labels else {}
        self.accumulator = _Accumulator()
        self.num_classes = num_classes
        self.ignore_index = ignore_index

    def metrics(self):
        metrics = self.accumulator.mean()
        metrics = _apply(
            lambda x, y: float(x),
            metrics,
            {}
        )
        metrics['mIOU'] = sum(
            [c['iou'] for _, c in metrics['class'].items()]
        ) / self.num_classes
        return metrics

    def _accuracy(self, outputs, targets):
        mask = (targets != self.ignore_index)
        num = torch.sum(outputs == targets).float()
        denom = torch.sum(mask)
        return num, denom

    def _iou(self, outputs, targets):
        mask = (targets != self.ignore_index).float()
        matches = (outputs == targets).float()
        ious = []
        for i in range(self.num_classes):
            num = (matches * (targets == i).float()).sum()
            union = (outputs == i).float() + (targets == i).float()
            intersection = (outputs == i).float() * (targets == i).float()
            denom = (mask * (union - intersection)).sum()
            ious.append((num, denom))
        return ious

    def add(self, outputs, targets, loss):
        metrics = {'loss': (loss, 1)}
        metrics['accuracy'] = self._accuracy(outputs, targets)
        iou = self._iou(outputs, targets)
        metrics['class'] = {
            self.labels.get(i, i): {
                'iou': iou[i],
            }
            for i in range(self.num_classes)}
        self.accumulator.update(metrics)


def compute_example_segmentation_metrics(
        n_classes, outputs, targets, loss, ignore_index=-1):
    metrics = TorchSegmentationMetrics(n_classes, ignore_index)
    metrics.add(outputs, targets, loss)
    return metrics.metrics()

