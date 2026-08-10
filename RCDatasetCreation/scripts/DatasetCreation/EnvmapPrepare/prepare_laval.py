"""
Prepare Laval Indoor HDR Dataset with exposure correction.

The original images in the Laval dataset are too dark (likely an EXR format
issue), so we apply automatic exposure adjustment before saving.
"""
import argparse
import cv2
import imageio
import numpy as np
from envmap_utils import collect_hdr_files, split_train_test, save_with_index


def process_laval_data(path, target_exposure=0.2):
    """Read an HDR image and adjust exposure for dark Laval dataset images."""
    img = imageio.v2.imread(path)
    mask = np.where(img.mean(axis=-1) < 1)
    mean = img[mask].mean()
    img *= target_exposure / mean
    img = cv2.resize(img, (img.shape[1] // 2, img.shape[0] // 2))
    return img


def main():
    parser = argparse.ArgumentParser(
        description='Prepare Laval Indoor HDR Dataset with exposure correction.')
    parser.add_argument('--input_folder', required=True,
                        help='Laval dataset folder containing .hdr/.exr files')
    parser.add_argument('--output_folder', required=True,
                        help='Root output directory')
    parser.add_argument('--test_percent', type=float, default=0.1,
                        help='Fraction of data reserved for test (default: 0.1)')
    parser.add_argument('--target_exposure', type=float, default=0.2,
                        help='Target mean exposure for dark region normalization (default: 0.2)')
    args = parser.parse_args()

    file_list = collect_hdr_files([args.input_folder])
    print(f'Collected {len(file_list)} envmaps from Laval dataset')

    train_list, test_list = split_train_test(file_list, args.test_percent)
    print(f'Train: {len(train_list)}, Test: {len(test_list)}')

    process_fn = lambda path: process_laval_data(path, args.target_exposure)
    save_with_index(train_list, args.output_folder, 'train', 'laval', process_fn=process_fn)
    save_with_index(test_list, args.output_folder, 'test', 'laval', process_fn=process_fn)


if __name__ == '__main__':
    main()
