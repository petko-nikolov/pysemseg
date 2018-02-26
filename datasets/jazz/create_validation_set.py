import argparse
import random
import shutil
import os

from utils import prompt_delete_dir
from datasets.jazz.jazz import parse_image_data


def copy_images(image_data, output_dir):
    for item in image_data:
        filepath_keys = [
            'image_filepath', 'ref_image_filepath',
            'mask_image_filepath']
        for filepath_key in filepath_keys:
            level = os.path.basename(os.path.dirname(item[filepath_key]))
            target_dir = os.path.join(output_dir, level)
            if not os.path.exists(target_dir):
                os.makedirs(target_dir)
            shutil.copyfile(
                item[filepath_key],
                os.path.join(
                    output_dir,
                    level,
                    os.path.basename(item[filepath_key])))


if '__main__' == __name__:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--input_dir', required=True, type=str,
        help='Input images path')
    parser.add_argument(
        '--output_train_dir', required=True, type=str,
        help='Outputh train path')
    parser.add_argument(
        '--output_val_dir', required=True, type=str,
        help='Output val path')
    parser.add_argument(
        '--percent_train', required=True, type=int,
        help='Input images path')
    args = parser.parse_args()

    prompt_delete_dir(args.output_train_dir)
    os.makedirs(args.output_train_dir)
    prompt_delete_dir(args.output_val_dir)
    os.makedirs(args.output_val_dir)

    image_data = parse_image_data(args.input_dir)
    random.shuffle(image_data)
    split_index = int(args.percent_train / 100 * len(image_data))
    copy_images(image_data[:split_index], args.output_train_dir)
    copy_images(image_data[split_index:], args.output_val_dir)
