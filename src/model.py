from typing import List

import torch
from torch import nn
from transformers import AutoModel

from src.constants import NUM_CLASSES


class MERTClassifier(nn.Module):
    def __init__(
        self,
        model_name: str = None,
        num_classes: int = None,
        hidden_states: str = 'first',
        thresholds: List[int] = None,
        pretrain: bool = True
    ) -> None:

        """
        MERTClassifier initializes a classification model based on a pretrained MERT model.
        
        Args:
            model_name (str): Name of the pretrained model.
            num_classes (int): Number of output classes for the classification.
            hidden_states (str, optional): Hidden states to use for classification. Defaults to 'first'.
            thresholds (list, optional): Thresholds for classification. Defaults to None.
            pretrained (bool, optional): Whether to freeze the pretrained MERT model's parameters. Defaults to True.
        """
        super(MERTClassifier, self).__init__()
        self.model_name = model_name
        self.num_classes = num_classes if num_classes is not None else NUM_CLASSES
        self.hidden_states = hidden_states
        self.thresholds = thresholds if thresholds is not None else [0.5] * self.num_classes
        self.mert_model = self.load_mert_model(pretrain)

        hidden_dim = self.mert_model.encoder.layers[-2].feed_forward.output_dense.out_features

        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(hidden_dim, self.num_classes)

    def load_mert_model(self, pretrain):
        """
        Loads a pretrained MERT model and optionally freezes its parameters.
        
        Args:
            model_name (str): Name of the pretrained model.
            freeze_pretrained (bool): Whether to freeze the pretrained model's parameters.
            
        Returns:
            Pretrained MERT model with frozen or unfrozen parameters.
        """
        mert_model = AutoModel.from_pretrained(self.model_name, trust_remote_code=True)
        if pretrain:
            for param in mert_model.parameters():
                param.requires_grad = False
        return mert_model

    def forward(self, x):
        """
        Forward pass through the MERT model.
        
        Args:
            x (Tensor): Input tensor for the model.
            
        Returns:
            Tensor: Output logits after passing through the classifier head.
        """
        outputs = self.mert_model(x.squeeze(1), output_hidden_states=True)

        if self.hidden_states == "all":
            hidden_states = torch.stack(outputs.hidden_states).permute(1, 0, 2, 3).mean(dim=1)
        elif self.hidden_states == "first":
            hidden_states = outputs[0]
        
        time_reduced_states = hidden_states.mean(dim=1).view(hidden_states.shape[0], -1)
        return self.fc(time_reduced_states)

    @torch.no_grad()
    def predict(self, x):
        """
        Perform predictions using sigmoid activation and thresholding.
        
        Args:
            x (Tensor): Input tensor for prediction.
            
        Returns:
            Tensor: Binary prediction tensor.
        """
        thresholds_tensor = torch.tensor(self.thresholds, device=x.device)
        probs = torch.sigmoid(self(x))
        return (probs > thresholds_tensor).int()
