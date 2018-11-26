import os
import numpy as np
import cv2
import visdom

from pysemseg.utils import flatten_dict
from pysemseg.transforms import ResizeBatch


IMAGES_WIDTH = 128

class VisdomLogger:
    def __init__(self, log_directory, color_palette, continue_logging=False):
        self.log_directory = log_directory
        self.color_palette = color_palette
        self.coninue_logging = continue_logging
        visdom_env = os.path.basename(log_directory)
        self.visdom = visdom.Visdom(
            env=visdom_env,
            log_to_filename=os.path.join(log_directory, 'viz.log')
        )
        if not continue_logging:
            self.visdom.delete_env(visdom_env)

    def log_args(self, args_dict):
        args_text = "<br />".join("{}: {}".format(k, v) for k, v in args_dict.items())
        self.visdom.text(
            text=args_text,
            win='Args',
        )

    def _update_metric_plots(self, iteration, metrics, prefix):
        for key, value in flatten_dict(metrics).items():
            name = "{}/{}".format(prefix, key)
            self.visdom.line(
                np.array([value]),
                np.array([iteration]),
                win=name,
                update='append' if iteration > 0 else None,
                opts={'title': name}
            )

    def _log_current_class_metrics(self, metrics, prefix):
        class_metric_names = list(next(iter(metrics['class'].values())).keys())
        for metric_name in class_metric_names:
            name = '{}/Current/{}'.format(prefix, metric_name)
            class_names = list(metrics['class'].keys())
            values = np.array([v[metric_name] for v in metrics['class'].values()])
            self.visdom.bar(
                values,
                win=name,
                opts={'rownames': class_names, 'title': name}
            )

    def log_metrics(self, iteration, metrics, prefix):
        self._update_metric_plots(iteration, metrics, prefix)
        self._log_current_class_metrics(metrics, prefix)

    def log_prediction_images(self, iteration, image, ground_truth, prediction, name, prefix):
        title = '{}/{}'.format(prefix, name)
        ground_truth = self.color_palette.encode_color(ground_truth)
        prediction = self.color_palette.encode_color(prediction)
        height = int(IMAGES_WIDTH / image.shape[3] * image.shape[2])
        image = ResizeBatch((height, IMAGES_WIDTH))(image.transpose([0, 2, 3, 1]))
        prediction = ResizeBatch(
            (height, IMAGES_WIDTH), interpolation=cv2.INTER_NEAREST)(prediction)
        ground_truth = ResizeBatch(
            (height, IMAGES_WIDTH), interpolation=cv2.INTER_NEAREST)(ground_truth)
        combined_images = np.concatenate([image, ground_truth, prediction], axis=2)
        self.visdom.images(
            combined_images.transpose([0, 3, 1, 2]),
            nrow=1,
            win=title,
            opts={'caption': title}
        )
