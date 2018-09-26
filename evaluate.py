import numpy as np
from torch.autograd import Variable
from metrics import SegmentationMetrics, flatten_metrics
from utils import tensor_to_numpy


def evaluate(
        model, loader, criterion, logger, epoch, summary_writer, cuda=False):
    model.eval()

    metrics = SegmentationMetrics(loader.dataset.number_of_classes)
    for _, (_, data, target) in enumerate(loader):
        if cuda:
            data, target = data.cuda(), target.cuda()

        data, target = Variable(data), Variable(target)
        output = model(data)
        loss = criterion(output, target)

        output = tensor_to_numpy(output.data)
        predictions = np.argmax(output, axis=1)

        metrics.add(
            predictions,
            tensor_to_numpy(target.data),
            float(tensor_to_numpy(loss.data)))

    metrics_dict = flatten_metrics(metrics.metrics())

    logger.log(
        len(loader), epoch, loader, data,
        metrics_dict, mode='Validation')

    if summary_writer is not None:
        for k, v in metrics_dict.items():
            summary_writer.add_scalar(
                "validation/{}".format(k), v, epoch)

    return predictions
