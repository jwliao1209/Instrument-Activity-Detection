import torch
from torch import nn
from transformers import AutoModel


class MERTClassifier(nn.Module):
    def __init__(self, hidden_dim=768, num_classes=9):
        super(MERTClassifier, self).__init__()
        self.load_pretrained_model()
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def load_pretrained_model(self):
        self.pretrained_model = AutoModel.from_pretrained(
            'm-a-p/MERT-v1-95M',
            trust_remote_code=True
        )
        for param in self.pretrained_model.parameters():
            param.requires_grad = False

    def forward(self, inputs):
        inputs['input_values'] = inputs['input_values'].squeeze(1)
        outputs = self.pretrained_model(**inputs, output_hidden_states=True)
        layer_hidden_states = torch.stack(outputs.hidden_states).permute(1, 0, 2, 3).mean(dim=1)
        time_reduced_hidden_states = layer_hidden_states.mean(dim=1).view(layer_hidden_states.shape[0], -1)
        return self.fc(time_reduced_hidden_states)
    
    def predict(self, inputs):
        return (torch.sigmoid(self.forward(inputs)) > 0.51).int()

