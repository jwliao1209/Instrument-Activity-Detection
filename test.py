import os
from argparse import ArgumentParser, Namespace

import numpy as np
import matplotlib.pyplot as plt
import torch
from sklearn.metrics import classification_report, roc_curve, auc
from tqdm import tqdm

from src.constants import CKPT_FILE, RESULT_DIR, CATEGORIES, NUM_CLASSES
from src.dataset import MusicDataset
from src.model import MERTClassifier
from src.transform import get_transforms
from src.utils import set_random_seeds, read_json


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
        default='checkpoints/10-02-04-36-30',
    )
    return parser.parse_args()


if __name__ == "__main__":
    set_random_seeds()
    args = parse_arguments()
    test_data = read_json(args.test_data_path)

    config = read_json(os.path.join(args.ckpt_dir, 'config.json'))

    transforms = get_transforms(model_name=config['model_name'])
    test_loader = MusicDataset(test_data, transforms).get_loader()

    # Prepare inference
    checkpoint = torch.load(os.path.join(args.ckpt_dir, CKPT_FILE), weights_only=True)
    device = torch.device(f'cuda:0'if torch.cuda.is_available() else 'cpu')
    model = MERTClassifier(
        model_name=config['model_name'],
        hidden_states=config['hidden_states'],
        fine_tune_method=config['fine_tune_method'],
    )
    model.load_state_dict(checkpoint['model'])
    model = model.to(device)
    model.eval()

    probas = []
    preds = []
    labels = []
    for data in tqdm(test_loader):
        proba = model.predict_proba(data['audio'].to(device)).cpu()
        pred =  (proba > 0.5).int()
        probas.append(proba)
        preds.append(pred)
        labels.append(data['label'])

    probas = torch.cat(probas).numpy()
    preds = torch.cat(preds).numpy()
    labels = torch.cat(labels).numpy()

    # Generate classification report
    report = classification_report(labels, preds, target_names=CATEGORIES)
    print(f"Classification report:\n{report}")


    # Plot ROC curve
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    thresholds = dict()
    best_thresholds = dict()

    plt.figure(figsize=(6, 6), dpi=600)
    for i in range(NUM_CLASSES):
        fpr[i], tpr[i], thresholds[i] = roc_curve(labels[:, i], probas[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

        # Calculus Youden's J statistic
        J = tpr[i] - fpr[i]
        ix = np.argmax(J)
        best_thresholds[i] = thresholds[i][ix]

        plt.plot(
            fpr[i], tpr[i],
            label=f'{CATEGORIES[i]} (AUC: {roc_auc[i]:.2f})'
        )

        print(i, best_thresholds[i])

    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve for Multi-Label Classification')
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(RESULT_DIR, 'roc_curve.png'))

    # Generate best classification report
    threshold = np.array([float(best_thresholds[i]) for i in range(NUM_CLASSES)]).reshape(1, -1)
    best_preds = (probas > threshold).astype('int')
    report = classification_report(labels, best_preds, target_names=CATEGORIES)
    print(f"Classification report with best thresholds: {threshold}\n{report}")
