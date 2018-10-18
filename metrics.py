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
    def __init__(self, num_classes):
        self.accumulator = _Accumulator()
        self.num_classes = num_classes

    def metrics(self):
        metrics = self.accumulator.mean()
        metrics['mIOU'] = np.mean(
            [c['iou'] for _, c in metrics['class'].items()]
        )
        return metrics

    @classmethod
    def _accuracy(cls, outputs, targets):
        num = np.sum(outputs == targets)
        denom = outputs.size
        return num, denom

    def _recall(self, outputs, targets):
        matches = (outputs == targets)
        return [
            (np.sum(matches[targets == i]), np.sum(targets == i))
            for i in range(self.num_classes)
        ]

    def _precision(self, outputs, targets):
        matches = (outputs == targets)
        return [(np.sum(matches[outputs == i]), np.sum(outputs == i))
                for i in range(self.num_classes)]

    def _iou(self, outputs, targets):
        matches = (outputs == targets)
        return [
            (np.sum(matches[targets == i]),
             np.sum(np.logical_or(outputs == i, targets == i)))
            for i in range(self.num_classes)
        ]

    def add(self, outputs, targets, loss):
        metrics = {'loss': (loss, 1)}
        metrics['accuracy'] = self._accuracy(outputs, targets)
        recall = self._recall(outputs, targets)
        precision = self._precision(outputs, targets)
        iou = self._iou(outputs, targets)
        metrics['class'] = {
            i: {
                'recall': recall[i],
                'precision': precision[i],
                'iou': iou[i],
            }
            for i in range(self.num_classes)}
        self.accumulator.update(metrics)


def compute_example_segmentation_metrics(n_classes, outputs, targets, loss):
    metrics = SegmentationMetrics(n_classes)
    metrics.add(outputs, targets, loss)
    return metrics.metrics()


def flatten_metrics(metrics):
    result = {}
    for key, value in metrics.items():
        if isinstance(value, dict):
            flattened = flatten_metrics(value)
            result.update({
                "{}/{}".format(key, kk): fv
                for kk, fv in flattened.items()})
        else:
            result[key] = value
    return result
