import pprint
import json


class StepLogger():

    def __init__(self, filename=None):
        self.filename = filename
        self.log_file = None

    def __enter__(self):
        self.log_file = open(self.filename, 'w')
        return self

    def __exit__(self, *args, **kwargs):
        self.log_file.close()

    def log(self, index, epoch, loader, data, metrics, mode='Train'):
        print("".join(40 * ['-']))
        print('{} Epoch: {} [{}/{} ({:.0f}%)]'.format(
            mode, epoch, index * len(data), len(loader.dataset),
            100. * index / len(loader)))
        pprint.pprint(metrics)
        if self.log_file is not None:
            log_data = {**metrics, 'epoch': epoch, 'step': index, "mode": mode}
            self.log_file.write(json.dumps(log_data) + '\n')
