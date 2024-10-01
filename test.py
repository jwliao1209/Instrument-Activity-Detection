import os
from argparse import ArgumentParser, Namespace

import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report
from tqdm import tqdm

from src.constants import CKPT_FILE, RESULT_DIR, CATEGORIES
from src.dataset import MusicDataset
from src.model import MERTClassifier
from src.transform import get_transforms
from src.utils import set_random_seeds, read_json, dict_to_device


def parse_arguments() -> Namespace:
    parser = ArgumentParser(description='Inference DL model')
    parser.add_argument(
        '--test_data_path',
        type=str,
        default='dataset/test.json',
    )
    parser.add_argument(
        '--ckpt_dir',
        type=str,
        default='checkpoints/10-01-21-18-42',
    )
    return parser.parse_args()


if __name__ == "__main__":
    set_random_seeds()
    args = parse_arguments()
    test_data = read_json(args.test_data_path)

    config = read_json(os.path.join(args.ckpt_dir, 'config.json'))

    transforms = get_transforms()
    test_loader = MusicDataset(test_data, transforms).get_loader()

    # Prepare inference
    checkpoint = torch.load(os.path.join(args.ckpt_dir, CKPT_FILE), weights_only=True)
    device = torch.device(f'cuda:0'if torch.cuda.is_available() else 'cpu')
    model = MERTClassifier(config.model_name)
    model.load_state_dict(checkpoint['model'])
    model = model.to(device)
    model.eval()

    preds = []
    labels = []
    with torch.no_grad():
        for data in tqdm(test_loader):
            data = dict_to_device(data, device)
            pred = model.predict(data['audio'])
            preds.append(pred)
            labels.append(data['label'])

    preds = torch.cat(preds).cpu().numpy()
    labels = torch.cat(labels).cpu().numpy()

    report = classification_report(labels, preds, target_names=CATEGORIES)
    print(report)
