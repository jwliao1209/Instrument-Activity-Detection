import json
import random
from datetime import datetime

import numpy as np
import torch


def read_json(path):
    with open(path, 'r') as f:
        return json.load(f)


def save_json(data, path):
    with open(path, 'w') as f:
        json.dump(data, f, indent=4)


def get_time():
    return datetime.today().strftime('%m-%d-%H-%M-%S')


def flatten_features(feature_dict):
    flatten_feature_dict = {}
    for feature_name, values in feature_dict.items():
        if isinstance(values, list):
            for i in range(len(values)):
                flatten_feature_dict[f'{feature_name}_{i}'] = values[i]
        else:
            flatten_feature_dict[feature_name] = values
    return flatten_feature_dict


def set_random_seeds(seed: int = 0) -> None:
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


def dict_to_device(data: dict, device: torch.device) -> dict:
    return {k: v.to(device) if not isinstance(v, list) else v for k, v in data.items()}
