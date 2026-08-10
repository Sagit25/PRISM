"""
Prepare BasicShape data for the dataset.

Copies pre-generated BasicShape meshes (poissonSubd.ply) from the raw dataset,
verifies they fit within a unit cube, and organizes them into train/test splits.
"""
import argparse
import glob
import os
import os.path as osp
import shutil
import numpy as np
import trimesh
from tqdm import tqdm


def check_mesh_size(data_folder, split, num_check=100):
    """Verify that all meshes have bounds within [-1, 1]."""
    shapes = glob.glob(osp.join(data_folder, split, 'Shape__*'))
    for shape_dir in shapes[:num_check]:
        mesh_path = osp.join(shape_dir, 'poissonSubd.ply')
        mesh = trimesh.load(mesh_path)
        assert np.all(np.abs(mesh.bounds) < 1.0), f'Mesh out of bounds: {mesh_path}'


def copy_split(data_folder, output_folder, split):
    """Copy meshes for a given split and write the index file."""
    data_path_list = glob.glob(osp.join(data_folder, split, 'Shape__*'))
    data_path_list.sort(key=lambda x: int(x.split('/')[-1].split('__')[-1]))

    split_folder = osp.join(output_folder, 'BasicShape', split)
    os.makedirs(split_folder, exist_ok=True)

    index_path = osp.join(output_folder, f'{split}_shape_basic.txt')
    with open(index_path, 'w') as f:
        for data_path in tqdm(data_path_list, desc=split):
            shape_id = data_path.split('/')[-1].split('__')[-1]
            src = osp.join(data_path, 'poissonSubd.ply')
            dst_name = f'BasicShape_{shape_id}.ply'
            dst = osp.join(split_folder, dst_name)
            shutil.copy2(src, dst)
            f.write(osp.join('BasicShape', split, dst_name) + '\n')


def main():
    parser = argparse.ArgumentParser(
        description='Prepare BasicShape data: copy and organize into train/test.')
    parser.add_argument('--input_folder', required=True,
                        help='Raw BasicShape folder (containing train/test subdirs with Shape__* folders)')
    parser.add_argument('--output_folder', required=True,
                        help='Root output directory for organized shapes')
    parser.add_argument('--check', action='store_true',
                        help='Run mesh size validation before copying')
    args = parser.parse_args()

    if args.check:
        print('Checking mesh sizes...')
        check_mesh_size(args.input_folder, 'train')
        check_mesh_size(args.input_folder, 'test')
        print('All meshes within bounds.')

    for split in ['train', 'test']:
        copy_split(args.input_folder, args.output_folder, split)


if __name__ == '__main__':
    main()
