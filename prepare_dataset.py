import os
from argparse import ArgumentParser, Namespace

import numpy as np

from src.constants import DATA_DIR
from src.utils import read_json, save_json


def parse_arguments() -> Namespace:
    parser = ArgumentParser(description='Prepare dataset')
    parser.add_argument(
        '--data_path',
        type=str,
        default='hw1/slakh/train_labels.json',
    )
    parser.add_argument(
        '--data_folder',
        type=str,
        default='hw1/slakh/train',
    )
    parser.add_argument(
        '--save_path',
        type=str,
        default='dataset/train.json',
    )
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_arguments()
    data_list = read_json(args.data_path)
    processed_data_list = []
    for filename, label in data_list.items():
        processed_data_list.append(
            {
                'audio': os.path.join(args.data_folder, filename),
                'label': label,
            }
        )
    os.makedirs(DATA_DIR, exist_ok=True)
    save_json(processed_data_list, args.save_path)
