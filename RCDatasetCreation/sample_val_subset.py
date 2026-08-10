"""
Generate a small validation subset by sampling from the full validation set.

Samples equally from BasicShape and OmniObject3D to create a balanced
small validation set for quick evaluation.
"""
import argparse
import random


def main():
    parser = argparse.ArgumentParser(
        description='Sample a small balanced validation set from the full val list.')
    parser.add_argument('--input', required=True,
                        help='Path to the full validation index file')
    parser.add_argument('--output', required=True,
                        help='Output path for the small validation index file')
    parser.add_argument('--total_num', type=int, default=200,
                        help='Total number of samples (split equally between BasicShape and Omniverse, default: 200)')
    args = parser.parse_args()

    with open(args.input) as f:
        all_lines = [x.strip() for x in f.readlines()]

    basic_lines = [x for x in all_lines if 'BasicShape' in x]
    omni_lines = [x for x in all_lines if 'BasicShape' not in x]

    half = args.total_num // 2
    sampled = sorted(random.sample(basic_lines, half)) + \
              sorted(random.sample(omni_lines, half))

    with open(args.output, 'w') as f:
        for line in sampled:
            f.write(line + '\n')

    print(f'Sampled {len(sampled)} entries ({half} BasicShape + {half} Omniverse)')


if __name__ == '__main__':
    main()
