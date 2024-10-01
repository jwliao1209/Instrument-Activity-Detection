from typing import List

import torch
from peft import LoraConfig, get_peft_model
from torch import nn
from transformers import AutoModel

from src.constants import NUM_CLASSES


class MERTClassifier(nn.Module):
    def __init__(
        self,
        model_name: str = None,
        num_classes: int = None,
        hidden_states: str = 'first',
        fine_tune_method: str = 'last_layer',
        thresholds: List[int] = None,
    ) -> None:

        super(MERTClassifier, self).__init__()
        self.model_name = model_name
        self.num_classes = num_classes if num_classes is not None else NUM_CLASSES
        self.hidden_states = hidden_states
        self.fine_tune_method = fine_tune_method
        self.mert_model = self.load_mert_model()
        self.fc = nn.Linear(
            self.mert_model.encoder.layers[-2].feed_forward.output_dense.out_features,
            self.num_classes
        )
        self.thresholds = thresholds if thresholds is not None else [0.5] * self.num_classes

    def load_mert_model(self):
        mert_model = AutoModel.from_pretrained(self.model_name, trust_remote_code=True)
        if self.fine_tune_method == 'last_layer':
            for param in mert_model.parameters():
                param.requires_grad = False
            print("Only fine-tune the last layer")

        elif self.fine_tune_method == 'lora':
            lora_config = LoraConfig(
                r=4,
                lora_alpha=8,
                lora_dropout=0.1,
                target_modules=['q_proj', 'v_proj'],
            )
            mert_model = get_peft_model(mert_model, lora_config)
            print("Use LoRA to fine-tune the model")

        else:  # full-parameters
            print("Fine-tune the full parameters")

        return mert_model

    def forward(self, x):
        outputs = self.mert_model(x.squeeze(1), output_hidden_states=True)
        if self.hidden_states == 'all':
            hidden_states = torch.stack(outputs.hidden_states).permute(1, 0, 2, 3).mean(dim=1)
        elif self.hidden_states == 'first':
            hidden_states = outputs[0]
        time_reduced_states = hidden_states.mean(dim=1).view(hidden_states.shape[0], -1)
        return self.fc(time_reduced_states)

    @torch.no_grad()
    def predict(self, x):
        thresholds_tensor = torch.tensor(self.thresholds, device=x.device)
        probs = torch.sigmoid(self(x))
        return (probs > thresholds_tensor).int()
