import numpy as np
from torch.autograd import Variable
import torch
from pysemseg.metrics import SegmentationMetrics
from pysemseg.utils import tensor_to_numpy, flatten_dict


def evaluate(
        model, loader, criterion, console_logger, epoch,
        visual_logger, device):
    model.eval()

    metrics = SegmentationMetrics(
        loader.dataset.number_of_classes,
        loader.dataset.labels,
        ignore_index=loader.dataset.ignore_index
    )

    with torch.no_grad():
        for step, (_, data, target) in enumerate(loader):
            data, target = data.to(device), target.to(device)

            data, target = Variable(data), Variable(target)
            output = model(data)
            loss = criterion(output, target)

            output, target, loss = [
                tensor_to_numpy(t.data) for t in [output, target, loss]
            ]

            predictions = np.argmax(output, axis=1)

            loss = loss / np.sum(target != loader.dataset.ignore_index)

            metrics.add(predictions, target, float(loss))

            if step % 10 == 0:
                visual_logger.log_prediction_images(
                    step,
                    tensor_to_numpy(data.data),
                    target.data,
                    predictions,
                    name='images',
                    prefix='Validation'
                )

    metrics_dict = metrics.metrics()

    console_logger.log(
        len(loader), epoch, loader, data,
        metrics_dict, mode='Validation')

    if visual_logger is not None:
        visual_logger.log_metrics(epoch, metrics_dict, prefix='Validation')

    return predictions
