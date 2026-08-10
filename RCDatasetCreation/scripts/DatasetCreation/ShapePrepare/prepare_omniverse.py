"""
Prepare OmniObject3D shapes for the dataset.

Pipeline (run as three separate subcommands):
1. select:   Filter shapes by category/object lists, normalize to unit cube,
             simplify to ~100k faces, repair holes, smooth, rotate to target coords.
2. simplify: Further simplify to ~10k faces with additional smoothing.
3. split:    Randomly split into train/test and generate index files.
"""
import argparse
import glob
import math
import os
import os.path as osp
import random
import shutil

import numpy as np
import pymeshlab
import trimesh
from tqdm import tqdm


class Selector:
    """Filter OmniObject3D meshes by coarse (category) and fine (object) name lists."""

    def __init__(self, coarse_file, fine_file):
        with open(coarse_file, 'r') as f:
            self.coarse_list = [line.strip() for line in f.readlines()]
        with open(fine_file, 'r') as f:
            self.fine_list = [line.strip() for line in f.readlines()]

    def __call__(self, coarse_name, fine_name):
        return coarse_name in self.coarse_list or fine_name in self.fine_list


def _iterative_repair_simplify(ms, target_facenum, max_iter=5):
    """Iteratively repair and simplify mesh until target face count is reached."""
    for _ in range(max_iter):
        ms.apply_filter('meshing_remove_duplicate_faces')
        ms.apply_filter('meshing_remove_duplicate_vertices')
        ms.apply_filter('meshing_repair_non_manifold_edges')
        ms.apply_filter('meshing_merge_close_vertices',
                        threshold=pymeshlab.Percentage(3))
        ms.apply_filter('meshing_repair_non_manifold_edges')
        ms.apply_filter('meshing_close_holes', maxholesize=30)
        face_num = ms.current_mesh().face_number()
        cur_target = max(target_facenum, face_num // 2)
        ms.apply_filter('meshing_decimation_quadric_edge_collapse',
                        targetfacenum=cur_target)
        if cur_target == target_facenum:
            break

    # Final cleanup
    ms.apply_filter('meshing_remove_duplicate_vertices')
    ms.apply_filter('meshing_repair_non_manifold_edges')
    ms.apply_filter('meshing_merge_close_vertices',
                    threshold=pymeshlab.Percentage(3))
    ms.apply_filter('meshing_repair_non_manifold_edges')
    ms.apply_filter('meshing_close_holes', maxholesize=30)


def select_and_simplify(raw_data_folder, output_folder, selector):
    """
    Select meshes from raw OmniObject3D, normalize to unit cube,
    simplify to ~100k faces, repair and smooth.
    """
    mesh_files = glob.glob(
        osp.join(raw_data_folder, '**', '**', 'Scan', 'Scan.obj'),
        recursive=True)
    mesh_files.sort()
    os.makedirs(output_folder, exist_ok=True)

    for filepath in tqdm(mesh_files, desc='select+simplify'):
        coarse_name = filepath.split('/')[-4]
        fine_name = filepath.split('/')[-3]
        if not selector(coarse_name, fine_name):
            continue

        output_path = osp.join(output_folder, fine_name + '.ply')
        if osp.exists(output_path):
            continue

        try:
            ms = pymeshlab.MeshSet()
            ms.load_new_mesh(filepath)

            # Normalize to unit cube
            bbox = ms.current_mesh().bounding_box()
            bounding_box = np.vstack([bbox.min(), bbox.max()])
            center = np.mean(bounding_box, axis=0)
            scale = 1.0 / (np.max(bounding_box[1] - bounding_box[0]) / 2 / 0.95)
            ms.apply_filter('compute_matrix_from_translation',
                            axisx=-center[0], axisy=-center[1], axisz=-center[2])
            ms.apply_filter('compute_matrix_from_translation_rotation_scale',
                            scalex=scale, scaley=scale, scalez=scale)

            _iterative_repair_simplify(ms, target_facenum=100000)
            ms.apply_filter('apply_coord_taubin_smoothing',
                            stepsmoothnum=5, lambda_=0.5)

            ms.save_current_mesh(output_path)

            # Rotate 90 degrees around X axis (OmniObject3D coordinate convention)
            mesh = trimesh.load(output_path)
            mesh.apply_transform(
                trimesh.transformations.rotation_matrix(math.pi / 2, [1, 0, 0]))
            mesh.export(output_path)
        except Exception:
            print(f'Error processing: {filepath}')
            continue


def further_simplify(input_folder, output_folder, target_facenum=10000):
    """Further simplify meshes to target face count with additional smoothing."""
    mesh_files = glob.glob(osp.join(input_folder, '*.ply'))
    mesh_files.sort()
    os.makedirs(output_folder, exist_ok=True)

    for filepath in tqdm(mesh_files, desc='further simplify'):
        filename = osp.basename(filepath)
        output_path = osp.join(output_folder, filename)

        ms = pymeshlab.MeshSet()
        ms.load_new_mesh(filepath)
        _iterative_repair_simplify(ms, target_facenum=target_facenum)
        ms.apply_filter('apply_coord_taubin_smoothing',
                        stepsmoothnum=10, lambda_=0.5)
        ms.save_current_mesh(output_path)

        # Re-export via trimesh to ensure clean format
        mesh = trimesh.load(output_path)
        mesh.export(output_path)


def split_data(input_folder, output_folder, test_num=76):
    """Split processed meshes into train/test and generate index files."""
    mesh_files = glob.glob(osp.join(input_folder, '*.ply'))
    random.shuffle(mesh_files)

    train_files = mesh_files[:len(mesh_files) - test_num]
    test_files = mesh_files[len(mesh_files) - test_num:]

    shape_dir = osp.join(output_folder, 'OmiverseShape')
    for split_name, file_list in [('train', train_files), ('test', test_files)]:
        split_dir = osp.join(shape_dir, split_name)
        os.makedirs(split_dir, exist_ok=True)

        index_path = osp.join(output_folder, f'{split_name}_shape_omiverse.txt')
        with open(index_path, 'w') as f:
            for filepath in tqdm(file_list, desc=split_name):
                basename = osp.basename(filepath)
                shutil.copy2(filepath, osp.join(split_dir, basename))
                f.write(f'OmiverseShape/{split_name}/{basename}\n')


def main():
    parser = argparse.ArgumentParser(
        description='Prepare OmniObject3D shapes: select, simplify, smooth, and split.')
    subparsers = parser.add_subparsers(dest='command', required=True)

    # Step 1: select and simplify from raw data
    p_select = subparsers.add_parser('select',
        help='Select and simplify meshes from raw OmniObject3D')
    p_select.add_argument('--raw_folder', required=True,
                          help='Raw OmniObject3D dataset folder')
    p_select.add_argument('--output_folder', required=True,
                          help='Output folder for selected+simplified meshes')
    p_select.add_argument('--coarse_list', required=True,
                          help='Path to coarse category selection list')
    p_select.add_argument('--fine_list', required=True,
                          help='Path to fine object selection list')

    # Step 2: further simplify
    p_simplify = subparsers.add_parser('simplify',
        help='Further simplify and smooth meshes')
    p_simplify.add_argument('--input_folder', required=True,
                            help='Folder with selected meshes from step 1')
    p_simplify.add_argument('--output_folder', required=True,
                            help='Output folder for further simplified meshes')
    p_simplify.add_argument('--target_facenum', type=int, default=10000,
                            help='Target face count (default: 10000)')

    # Step 3: split into train/test
    p_split = subparsers.add_parser('split',
        help='Split processed meshes into train/test')
    p_split.add_argument('--input_folder', required=True,
                         help='Folder with fully processed meshes')
    p_split.add_argument('--output_folder', required=True,
                         help='Root output directory for final dataset')
    p_split.add_argument('--test_num', type=int, default=76,
                         help='Number of test samples (default: 76)')

    args = parser.parse_args()

    if args.command == 'select':
        selector = Selector(args.coarse_list, args.fine_list)
        select_and_simplify(args.raw_folder, args.output_folder, selector)
    elif args.command == 'simplify':
        further_simplify(args.input_folder, args.output_folder, args.target_facenum)
    elif args.command == 'split':
        split_data(args.input_folder, args.output_folder, args.test_num)


if __name__ == '__main__':
    main()
