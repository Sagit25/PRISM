"""Entry point for batch dataset generation."""
import argparse
import os

os.environ.setdefault('OPENCV_IO_ENABLE_OPENEXR', '1')

def parse_args():
    parser = argparse.ArgumentParser(description='Batch dataset generator')
    parser.add_argument('--conf', type=str, required=True,
                        help='Path to a PRISM dataset YAML config')
    parser.add_argument('--preview', action='store_true', help='Enable preview mode')
    parser.add_argument('--project_name', type=str, default='', help='Override project name')
    parser.add_argument('--output-folder', type=str, default='',
                        help='Override the dataset output root')
    parser.add_argument('--splits', nargs='+', choices=['train', 'validation', 'test'],
                        help='Generate only the selected dataset splits')
    parser.add_argument('--shard-count', type=int,
                        help='Number of deterministic shape shards')
    parser.add_argument('--shard-index', type=int,
                        help='Zero-based shard index')
    parser.add_argument('--device', type=str, choices=['cpu', 'gpu'], default='gpu',
                        help='Rendering backend device')
    return parser.parse_args()


def main():
    args = parse_args()

    import mitsuba as mi

    from projects import build_project
    from utils.config_utils import load_config
    from utils.tool_utils import set_random_seed

    variant_map = {
        'cpu': 'llvm_ad_rgb',
        'gpu': 'cuda_ad_rgb',
    }
    mi.set_variant(variant_map[args.device])
    set_random_seed(0)

    conf = load_config(args.conf)
    conf['preview'] = args.preview

    if args.project_name:
        conf['project_name'] = args.project_name
    if args.output_folder:
        conf['output_folder'] = args.output_folder
    if args.splits:
        conf['RunSplits'] = list(args.splits)
    if args.shard_count is not None or args.shard_index is not None:
        shard_count = 1 if args.shard_count is None else args.shard_count
        shard_index = 0 if args.shard_index is None else args.shard_index
        if shard_count < 1:
            raise ValueError('--shard-count must be at least one')
        if not 0 <= shard_index < shard_count:
            raise ValueError('--shard-index must be in [0, shard-count)')
        conf['Shard'] = {
            'enabled': shard_count > 1,
            'count': shard_count,
            'index': shard_index,
            'splits': list(args.splits or ['train']),
        }

    project = build_project(conf)
    os.makedirs(project.output_folder, exist_ok=True)
    project.run()


if __name__ == '__main__':
    main()
