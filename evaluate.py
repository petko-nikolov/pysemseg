from torch.autograd import Variable
from metrics import SegmentationMetrics, flatten_metrics


def evaluate(
        model, loader, criterion, logger, epoch, summary_writer,
        cuda=False):
    model.eval()

    metrics = SegmentationMetrics(2)
    for _, (data, target) in enumerate(loader):
        if cuda:
            data, target = data.cuda(), target.cuda()

        data, target = Variable(data), Variable(target)
        output = model(data)
        loss = criterion(output, target)

        output = output.data.numpy()
        target = target.data.numpy()

        metrics.add(
            output,
            target,
            float(loss.data.numpy()[0]))

    metrics_dict = flatten_metrics(metrics.metrics())

    logger.log(
        len(loader), epoch, loader, data,
        metrics_dict, mode='Validation')

    for k, v in metrics_dict.items():
        summary_writer.add_scalar(
            "validation/{}".format(k), v, epoch)
