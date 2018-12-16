import numpy as np
from torch.autograd import Variable
import torch
from pysemseg.metrics import TorchSegmentationMetrics
from pysemseg.utils import tensor_to_numpy, flatten_dict


def evaluate(
        model, loader, criterion, console_logger, epoch,
        visual_logger, device, log_images_interval):
    model.eval()

    metrics = TorchSegmentationMetrics(
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

            predictions = torch.argmax(output, dim=1)

            loss = loss / torch.sum(target != loader.dataset.ignore_index).float()

            metrics.add(predictions, target, loss)

            if step % log_images_interval == 0:
                visual_logger.log_prediction_images(
                    step,
                    tensor_to_numpy(data.data),
                    tensor_to_numpy(target.data),
                    tensor_to_numpy(predictions),
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
