"""
Prepare environment maps from PolyHeaven and similar HDR sources.

Note: The envmaps from OpenRooms and ambientCG are mostly outdoor scenes
containing large textureless sky areas, which are not suitable for training.
Only PolyHeaven and custom collections are used by default.
"""
import argparse
from envmap_utils import collect_hdr_files, split_train_test, save_with_index


def main():
    parser = argparse.ArgumentParser(
        description='Prepare PolyHeaven environment maps for train/test split.')
    parser.add_argument('--input_folders', nargs='+', required=True,
                        help='One or more folders containing .hdr/.exr files')
    parser.add_argument('--output_folder', required=True,
                        help='Root output directory')
    parser.add_argument('--test_percent', type=float, default=0.1,
                        help='Fraction of data reserved for test (default: 0.1)')
    parser.add_argument('--prefix', default='polyheaven',
                        help='Filename prefix for renamed files (default: polyheaven)')
    args = parser.parse_args()

    file_list = collect_hdr_files(args.input_folders)
    print(f'Collected {len(file_list)} envmaps from {len(args.input_folders)} folder(s)')

    train_list, test_list = split_train_test(file_list, args.test_percent)
    print(f'Train: {len(train_list)}, Test: {len(test_list)}')

    save_with_index(train_list, args.output_folder, 'train', args.prefix)
    save_with_index(test_list, args.output_folder, 'test', args.prefix)


if __name__ == '__main__':
    main()
