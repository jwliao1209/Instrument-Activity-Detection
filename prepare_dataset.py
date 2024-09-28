import os

import numpy as np

from src.utils import read_json, save_json


data_list = read_json('hw1/slakh/test_labels.json')

processed_data_list = []
for filename, label in data_list.items():
    processed_data_list.append(
        {
            'audio': f'hw1/slakh/test/{filename}',
            'label': label,
        }
    )

os.makedirs('dataset', exist_ok=True)
save_json(processed_data_list, 'dataset/test.json')
