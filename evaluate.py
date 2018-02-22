import numpy as np
from torch.autograd import Variable
from metrics import SegmentationMetrics, flatten_metrics
import tqdm


def evaluate(
        model, loader, criterion, logger, epoch, summary_writer, cuda=False):
    model.eval()

    metrics = SegmentationMetrics(loader.dataset.number_of_classes)
    for _, (data, target) in enumerate(loader):
        if cuda:
            data, target = data.cuda(), target.cuda()

        data, target = Variable(data), Variable(target)
        output = model(data)
        loss = criterion(output, target)

        output = output.data.numpy()
        predictions = np.argmax(output, axis=2)

        metrics.add(
            predictions,
            target.data.numpy(),
            float(loss.data.numpy()[0]))

    metrics_dict = flatten_metrics(metrics.metrics())

    logger.log(
        len(loader), epoch, loader, data,
        metrics_dict, mode='Validation')

    if summary_writer is not None:
        for k, v in metrics_dict.items():
            summary_writer.add_scalar(
                "validation/{}".format(k), v, epoch)

    return predictions

