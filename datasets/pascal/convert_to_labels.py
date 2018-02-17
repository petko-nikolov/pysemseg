import argparse
from multiprocessing import Pool, cpu_count
import numpy as np
import os
import glob
from transforms.loaders import CV2ImageLoader
import cv2
from tqdm import tqdm

from utils import prompt_delete_dir

output_dir = None

PASCAL_CLASSES = {
    'background': 0,
    'aeroplane': 1,
    'bicycle': 2,
    'bird': 3,
    'boat': 4,
    'bottle': 5,
    'bus': 6,
    'car': 7,
    'cat': 8,
    'chair': 9,
    'cow': 10,
    'diningtable': 11,
    'dog': 12,
    'horse': 13,
    'motorbike': 14,
    'person': 15,
    'potted-plant': 16,
    'sheep': 17,
    'sofa': 18,
    'train': 19,
    'tv/monitor': 20
}


def _get_palette_map(n):
    def bitget(byteval, idx):
        return ((byteval & (1 << idx)) != 0)
    color_to_label = {}
    for k in range(0, n):
        r = g = b = 0
        c = k
        for j in range(8):
            r = r | (bitget(c, 2) << 7 - j)
            g = g | (bitget(c, 1) << 7 - j)
            b = b | (bitget(c, 0) << 7 - j)
            c = c >> 3
        color_to_label[(r, g, b)] = k
    return color_to_label


PALETTE_MAP = _get_palette_map(256)


def process_mask(filename):
    basename = os.path.basename(filename)
    palette_mask = CV2ImageLoader()(filename)
    label_mask = np.zeros(palette_mask.shape[:2], dtype=np.uint8)
    for i in range(palette_mask.shape[0]):
        for j in range(palette_mask.shape[1]):
            label_mask[i, j] = PALETTE_MAP[tuple(palette_mask[i, j].tolist())]
    cv2.imwrite(os.path.join(output_dir, basename), label_mask)

if __name__ == '__main__':
    parser = argparse.ArgumentParser("Convert palette to labels")
    parser.add_argument("--input_dir", required=True, type=str)
    parser.add_argument("--output_dir", required=True, type=str)
    parser.add_argument("--ncpus", required=False, type=int)
    args = parser.parse_args()

    output_dir = args.output_dir
    prompt_delete_dir(output_dir)
    os.makedirs(output_dir)

    pool = Pool(processes=args.ncpus or cpu_count())

    files = glob.glob(os.path.join(args.input_dir, '*.png'))
    with tqdm(total=len(files)) as pbar:
        for i, _ in enumerate(pool.imap_unordered(process_mask, files)):
            pbar.update()
