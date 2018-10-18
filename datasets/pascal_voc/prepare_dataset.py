import os
import argparse
import glob
import shutil
from multiprocessing import Pool, cpu_count
import scipy.io
import cv2
from tqdm import tqdm
import numpy as np
from transforms.loaders import CV2ImageLoader

from utils import prompt_delete_dir


OUTPUT_DIR = None


def _get_palette_map(n_classes):
    def bitget(byteval, idx):
        return (byteval & (1 << idx)) != 0
    color_to_label = {}
    for k in range(0, n_classes):
        red = green = blue = 0
        cls = k
        for j in range(8):
            red = red | (bitget(cls, 2) << 7 - j)
            green = green | (bitget(cls, 1) << 7 - j)
            blue = blue | (bitget(cls, 0) << 7 - j)
            cls = cls >> 3
        color_to_label[(red, green, blue)] = k
    return color_to_label


COLOR_TO_LABEL = _get_palette_map(256)
LABEL_TO_COLOR = {v: k for k, v in COLOR_TO_LABEL.items()}


def encode_color(label_mask):
    if len(label_mask.shape) == 2:
        label_mask = np.expand_dims(label_mask, axis=-1)
    palette_mask = np.zeros(label_mask.shape[:2] + (3,), dtype=np.uint8)
    for i, mask in LABEL_TO_COLOR.items():
        palette_mask[np.where(np.all(label_mask == i, axis=-1))[:2]] = mask
    return palette_mask


def decode_color(palette_mask):
    label_mask = np.zeros(palette_mask.shape[:2], dtype=np.uint8)
    for mask, i in COLOR_TO_LABEL.items():
        label_mask[np.where(np.all(palette_mask == mask, axis=-1))[:2]] = i
    label_mask[label_mask == 255] = 0
    return label_mask


def process_mask(filename):
    basename = os.path.basename(filename)
    palette_mask = CV2ImageLoader()(filename)
    label_mask = decode_color(palette_mask)
    cv2.imwrite(os.path.join(OUTPUT_DIR, basename), label_mask)


def process_berkley_gt(filename):
    basename = os.path.basename(filename)
    image_id, _ = basename.split('.')
    mat = scipy.io.loadmat(
        filename, mat_dtype=True, squeeze_me=True, struct_as_record=False
    )
    segmentation = mat['GTcls'].Segmentation
    segmentation[segmentation == 255] = 0
    cv2.imwrite(
        os.path.join(OUTPUT_DIR, image_id + '.png'), mat['GTcls'].Segmentation
    )


def init_process(output_dir):
    global OUTPUT_DIR
    OUTPUT_DIR = output_dir

def convert_voc2012_labels(input_dir, output_dir, ncpus=None, overwrite=False):
    if overwrite:
        shutil.rmtree(output_dir, ignore_errors=True)
    else:
        prompt_delete_dir(output_dir)

    os.makedirs(output_dir)

    files = glob.glob(os.path.join(input_dir, '*.png'))

    with Pool(processes=ncpus or cpu_count(),
              initializer=init_process, initargs=(output_dir,)) as pool:
        with tqdm(total=len(files), desc='Prepare VOC2012') as pbar:
            for _ in enumerate(pool.imap_unordered(process_mask, files)):
                pbar.update()



def convert_berkley_labels(input_dir, output_dir, ncpus=None, overwrite=False):
    if overwrite:
        shutil.rmtree(output_dir, ignore_errors=True)
    else:
        prompt_delete_dir(output_dir)
    os.makedirs(output_dir)

    files = glob.glob(os.path.join(input_dir, '*.mat'))

    with Pool(processes=ncpus or cpu_count(),
              initializer=init_process, initargs=(output_dir,)) as pool:
        with tqdm(total=len(files), desc='Prepare Berkley') as pbar:
            for _ in enumerate(pool.imap_unordered(
                    process_berkley_gt, files)):
                pbar.update()


def prepare_dataset(rootdir, ncpus=None, overwrite=False):
    voc2012_root = os.path.join(rootdir, 'VOCdevkit/VOC2012')
    berkley_root = os.path.join(rootdir, 'benchmark_RELEASE')
    assert os.path.exists(voc2012_root)
    convert_voc2012_labels(
        os.path.join(voc2012_root, 'SegmentationClass'),
        os.path.join(voc2012_root, 'SegmentationClassLabels'),
        ncpus,
        overwrite)
    convert_berkley_labels(
        os.path.join(berkley_root, 'dataset/cls/'),
        os.path.join(berkley_root, 'dataset/cls_labels'),
        ncpus,
        overwrite)


def main():
    parser = argparse.ArgumentParser("Convert palette to labels")
    parser.add_argument("--dataset_dir", required=True, type=str)
    parser.add_argument("--overwrite", action='store_true', default=False,
                        help="Overwrite existing data.")
    parser.add_argument("--ncpus", required=False, type=int)
    args = parser.parse_args()
    prepare_dataset(args.dataset_dir, args.ncpus, args.overwrite)


if __name__ == '__main__':
    main()
