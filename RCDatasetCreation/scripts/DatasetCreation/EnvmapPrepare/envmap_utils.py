import glob
import os
import os.path as osp
import random
import shutil
from tqdm import tqdm


def collect_hdr_files(folders):
    """Collect all HDR/EXR files from a list of folders."""
    file_list = []
    for folder in folders:
        file_list.extend(glob.glob(osp.join(folder, '*.hdr')))
        file_list.extend(glob.glob(osp.join(folder, '*.exr')))
    return file_list


def split_train_test(file_list, test_percent=0.1):
    """Randomly shuffle and split file list into train/test sets."""
    file_list = list(file_list)
    random.shuffle(file_list)
    split_idx = int(len(file_list) * (1 - test_percent))
    return file_list[:split_idx], file_list[split_idx:]


def save_with_index(file_list, output_folder, split_name, prefix, process_fn=None):
    """
    Copy or process files into output_folder/split_name/ and write an index file.

    Args:
        file_list: list of source file paths
        output_folder: root output directory
        split_name: 'train' or 'test'
        prefix: filename prefix for renamed files (e.g. 'polyheaven', 'laval')
        process_fn: optional callable(path) -> image array. If provided, the
                    returned image is saved via imageio instead of copying.
    """
    split_folder = osp.join(output_folder, split_name)
    os.makedirs(split_folder, exist_ok=True)

    index_file = osp.join(output_folder, f'{split_name}_env.txt')
    with open(index_file, 'w') as f:
        for idx, src_path in enumerate(tqdm(file_list, desc=split_name)):
            ext = src_path.split('.')[-1]
            dst_name = f'{prefix}_{idx}.{ext}'
            dst_path = osp.join(split_folder, dst_name)

            if process_fn is not None:
                import imageio
                img = process_fn(src_path)
                imageio.imwrite(dst_path, img)
            else:
                shutil.copy2(src_path, dst_path)

            f.write(osp.join(split_name, dst_name) + '\n')
