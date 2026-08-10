"""Entry point for batch dataset generation."""
import argparse
import os

os.environ.setdefault('OPENCV_IO_ENABLE_OPENEXR', '1')

def parse_args():
    parser = argparse.ArgumentParser(description='Batch dataset generator')
    parser.add_argument('--conf', type=str, default='configs/dataset.yaml',
                        help='Path to the .yaml config file')
    parser.add_argument('--preview', action='store_true', help='Enable preview mode')
    parser.add_argument('--project_name', type=str, default='', help='Override project name')
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

    project = build_project(conf)
    os.makedirs(project.output_folder, exist_ok=True)
    project.run()


if __name__ == '__main__':
    main()
