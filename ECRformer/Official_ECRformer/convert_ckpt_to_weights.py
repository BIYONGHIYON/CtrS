from argparse import ArgumentParser
from pathlib import Path

import torch

from util.checkpoint import (
    default_weights_output_path,
    describe_checkpoint,
    extract_state_dict,
    load_checkpoint_file,
)


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('input_ckpt', help='Path to the source checkpoint file')
    parser.add_argument('-o', '--output', default=None,
                        help='Output path for the weights-only checkpoint')
    parser.add_argument('--overwrite', action='store_true',
                        help='Overwrite the output file if it already exists')
    return parser.parse_args()


def main():
    args = parse_args()
    checkpoint = load_checkpoint_file(args.input_ckpt, map_location='cpu')
    summary = describe_checkpoint(checkpoint)
    state_dict = extract_state_dict(checkpoint)

    output_path = args.output or default_weights_output_path(args.input_ckpt)

    print('Checkpoint analysis:')
    print(f"  format: {summary['format']}")
    print(f"  top-level keys: {summary['top_level_keys']}")
    print(f"  number of tensors: {summary['num_tensors']}")
    print(f"  sample parameter keys: {summary['sample_state_keys']}")
    print(f'Output path: {output_path}')

    output_path = Path(output_path)
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(
            f'Output file already exists: {output_path}. Use --overwrite to replace it.'
        )

    torch.save(state_dict, output_path)
    print('Saved weights-only checkpoint successfully.')


if __name__ == '__main__':
    main()