import torch
import torch.nn as nn


class FocalBCEWithLogitsLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalBCEWithLogitsLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.bce_loss = nn.BCEWithLogitsLoss(reduction='none')
        self.reduction = reduction

    def forward(self, x, y):
        bce_loss = self.bce_loss(x, y)
        probas = torch.sigmoid(x)
        focal_factor = torch.where(y == 1, (1 - probas) ** self.gamma, probas ** self.gamma)
        loss = focal_factor * bce_loss

        if self.alpha is not None:
            alpha_factor = torch.where(y == 1, self.alpha, 1 - self.alpha)
            loss = alpha_factor * loss

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


def get_loss(name: str) -> nn.Module:
    if name == 'bce_with_logits':
        return nn.BCEWithLogitsLoss()
    elif name == 'focal_bce_with_logits':
        return FocalBCEWithLogitsLoss()
    else:
        raise ValueError(f"Loss function {name} not found")
