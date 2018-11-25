import numpy as np


class _Accumulator():
    def __init__(self):
        self.numerator = {}
        self.denominator = {}

    def _apply(self, fcn, acc, counter):
        result = {}
        for key, value in acc.items():
            if isinstance(value, dict):
                result[key] = self._apply(fcn, value, counter.get(key, {}))
            else:
                result[key] = fcn(acc[key], counter.get(key, 0.0))
        return result

    def mean(self):
        return self._apply(
            lambda n, d: n / d if d > 0 else 0.0,
            self.numerator, self.denominator)

    def update(self, data):
        self.numerator = self._apply(
            lambda a, d: a[0] + d, data, self.numerator)
        self.denominator = self._apply(
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
                # 'recall': recall[i],
                # 'precision': precision[i],
                'iou': iou[i],
            }
            for i in range(self.num_classes)}
        self.accumulator.update(metrics)


def compute_example_segmentation_metrics(
        n_classes, outputs, targets, loss, ignore_index=-1):
    metrics = SegmentationMetrics(n_classes, ignore_index)
    metrics.add(outputs, targets, loss)
    return metrics.metrics()

