import numpy as np
from torch.autograd import Variable
from metrics import SegmentationMetrics
from utils import tensor_to_numpy, flatten_dict


def evaluate(
        model, loader, criterion, console_logger, epoch, visual_logger,
        cuda=False):
    model.eval()

    metrics = SegmentationMetrics(
        loader.dataset.number_of_classes, ignore_index=255)
    for step, (_, data, target) in enumerate(loader):
        if cuda:
            data, target = data.cuda(), target.cuda()

        data, target = Variable(data), Variable(target)
        output = model(data)
        loss = criterion(output, target)

        output = tensor_to_numpy(output.data)
        predictions = np.argmax(output, axis=1)

        mean_loss = loss / int(np.prod(target.shape))

        metrics.add(
            predictions,
            tensor_to_numpy(target.data),
            float(tensor_to_numpy(mean_loss.data)))

        if step % 10:
            visual_logger.log_prediction_images(
                step,
                tensor_to_numpy(data.data),
                tensor_to_numpy(target.data),
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
