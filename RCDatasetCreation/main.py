"""Entry point for the simplified renderer."""
import argparse
import os
import shutil

def parse_args():
    parser = argparse.ArgumentParser(description='Minimal Mitsuba renderer runner')
    parser.add_argument('--conf', type=str, default='configs/recon_data.yaml', help='Path to the .yaml config file')
    parser.add_argument('--preview', action='store_true', help='Enable preview mode overrides')
    parser.add_argument('--project_name', type=str, default='', help='Override project name in output path')
    parser.add_argument('--device', type=str, choices=['cpu', 'gpu'], default='gpu', help='Rendering backend device')
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
    if args.preview:
        conf['preview'] = True
    if args.project_name:
        conf['project_name'] = args.project_name

    project = build_project(conf)
    os.makedirs(project.output_folder, exist_ok=True)
    shutil.copyfile(args.conf, os.path.join(project.output_folder, 'config.yaml'))
    project.run()


if __name__ == '__main__':
    main()
