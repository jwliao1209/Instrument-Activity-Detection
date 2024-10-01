import os
from argparse import ArgumentParser, Namespace

import torch
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
        thresholds=[
            0.85112745,
            0.111269526,
            0.17468038,
            0.7682016,
            0.86039627,
            0.5402954,
            0.13542163,
            0.43342066,
            0.21783678,
        ]
    )
    model.load_state_dict(checkpoint['model'])
    model = model.to(device)
    model.eval()

    probas = []
    preds = []
    labels = []
    for data in tqdm(test_loader):
        data = dict_to_device(data, device)
        proba = model.predict_proba(data['audio'])
        pred = model.predict(data['audio'])

        probas.append(proba)
        preds.append(pred)
        labels.append(data['label'])

    probas = torch.cat(probas).cpu().numpy()
    preds = torch.cat(preds).cpu().numpy()
    labels = torch.cat(labels).cpu().numpy()

    report = classification_report(labels, preds, target_names=CATEGORIES)
    print(report)



    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve, auc


    n_classes = 9


    y_true = labels
    y_score = probas


    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    thresholds = dict()
    best_thresholds = dict()

    plt.figure()
    for i in range(n_classes):
        fpr[i], tpr[i], thresholds[i] = roc_curve(y_true[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

        # 計算 Youden's J statistic
        J = tpr[i] - fpr[i]
        ix = np.argmax(J)
        best_thresholds[i] = thresholds[i][ix]

        plt.plot(fpr[i], tpr[i], label=f'Class {i} (Best Threshold = {best_thresholds[i]:.2f})')

        print(i, best_thresholds[i])

    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve for Multi-Label Classification')
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(RESULT_DIR, 'roc_curve.png'))
