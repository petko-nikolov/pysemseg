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

from utils import prompt_delete_dir, ColorPalette


OUTPUT_DIR = None
color_palette = ColorPalette(256)


def process_mask(filename):
    basename = os.path.basename(filename)
    palette_mask = CV2ImageLoader()(filename)
    label_mask = color_palette.decode_color(palette_mask)
    cv2.imwrite(os.path.join(OUTPUT_DIR, basename), label_mask)


def process_berkley_gt(filename):
    basename = os.path.basename(filename)
    image_id, _ = basename.split('.')
    mat = scipy.io.loadmat(
        filename, mat_dtype=True, squeeze_me=True, struct_as_record=False
    )
    segmentation = mat['GTcls'].Segmentation
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
