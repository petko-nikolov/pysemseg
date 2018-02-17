import numpy as np


class _Accumulator():
    def __init__(self):
        self.accumulator = {}
        self.counter = {}

    def _apply(self, fn, acc, counter):
        result = {}
        for k, v in acc.items():
            if isinstance(v, dict):
                result[k] = self._apply(fn, v, counter.get(k, {}))
            else:
                result[k] = fn(acc[k], counter.get(k, 0.0))
        return result

    def mean(self):
        return self._apply(
            lambda a, c: a / c if c > 0 else 0.0,
            self.accumulator, self.counter)

    def update(self, data):
        self.accumulator = self._apply(
            lambda a, d: a + d, data, self.accumulator)
        self.counter = self._apply(lambda a, d: d + 1,
                                   data, self.counter)


class SegmentationMetrics:
    def __init__(self, num_classes):
        self.accumulator = _Accumulator()
        self.num_classes = num_classes

    def metrics(self):
        return self.accumulator.mean()

    def accuracy(self, outputs, targets):
        return np.mean(outputs == targets)

    def recall(self, outputs, targets):
        matches = (outputs == targets)
        mean_fn = lambda x: np.nan_to_num(np.mean(x))
        return [mean_fn(matches[targets == i])
                for i in range(self.num_classes)]

    def precision(self, outputs, targets):
        matches = (outputs == targets)
        mean_fn = lambda x: np.nan_to_num(np.mean(x))
        return [mean_fn(matches[outputs == i])
                for i in range(self.num_classes)]

    def iou(self, outputs, targets):
        matches = (outputs == targets)
        return [
            np.nan_to_num(np.sum(matches[targets == i]) / np.sum(
                np.logical_or(outputs == i, targets == i)))
            for i in range(self.num_classes)
        ]

    def add(self, outputs, targets, loss):
        metrics = {'loss': loss}
        metrics['accuracy'] = self.accuracy(outputs, targets)
        recall = self.recall(outputs, targets)
        precision = self.precision(outputs, targets)
        iou = self.iou(outputs, targets)
        metrics['class'] = {
            i: {
                'recall': recall[i],
                'precision': precision[i], 'iou': iou[i],
                'f1': 2 * recall[i] * precision[i] / (recall[i] + precision[i])
            }
            for i in range(self.num_classes)}
        self.accumulator.update(metrics)
        return metrics


def flatten_metrics(metrics):
    result = {}
    for k, v in metrics.items():
        if isinstance(v, dict):
            flattened = flatten_metrics(metrics[k])
            result.update({
                "{}/{}".format(k, kk): fv
                for kk, fv in flattened.items()})
        else:
            result[k] = v
    return result
