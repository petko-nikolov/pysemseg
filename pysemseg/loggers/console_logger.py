import json
from pysemseg.utils import flatten_dict


class ConsoleLogger():

    def __init__(self, filename=None, continue_training=False):
        self.filename = filename
        self.log_file = None
        self.continue_training = continue_training

    def __enter__(self):
        mode = 'w'
        if self.continue_training:
            mode += 'a'
        self.log_file = open(self.filename, mode)
        return self

    def __exit__(self, *args, **kwargs):
        self.log_file.close()

    def log(self, index, epoch, loader, data, metrics, mode='Train'):
        metric_str = ", ".join([
            '{}:{:.6f}'.format(k, v) for k, v in flatten_dict(metrics).items()
        ])
        print('{} Epoch: {} [{}/{} ({:.0f}%)] [{}]'.format(
            mode, epoch, index * len(data), len(loader.dataset),
            100. * index / len(loader), metric_str))
        if self.log_file is not None:
            log_data = {**metrics, 'epoch': epoch, 'step': index, "mode": mode}
            self.log_file.write(json.dumps(log_data) + '\n')
