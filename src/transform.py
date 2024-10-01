from functools import partial

import torch
import numpy as np
from torchvision.transforms import Compose
from transformers import Wav2Vec2FeatureExtractor

from src.constants import SAMPLE_RATE


class BaseTransform(object):
    def __init__(self, keys, **kwargs):
        self.keys = keys
        self._parse_var(**kwargs)

    def __call__(self, data, **kwargs):
        for key in self.keys:
            if key in data:
                data[key] = self._process(data[key], **kwargs)
            else:
                raise KeyError(f"{key} is not a key in data.")
        return data

    def _parse_var(self, **kwargs):
        pass

    def _process(self, single_data, **kwargs):
        NotImplementedError


class LoadAudio(BaseTransform):
    def __init__(self, keys, **kwargs):
        super(LoadAudio, self).__init__(keys, **kwargs)

    def _process(self, single_data, **kwargs):
        single_data = np.load(single_data)
        return single_data.reshape(1, *single_data.shape)


class ProcessAudio(BaseTransform):
    def __init__(self, keys, model_name, **kwargs):
        super(ProcessAudio, self).__init__(keys, **kwargs)
        processor = Wav2Vec2FeatureExtractor.from_pretrained(
            model_name, trust_remote_code=True,
        )
        self.processor = partial(
            processor, sampling_rate=SAMPLE_RATE, return_tensors='pt', padding=True,
        )

    def _process(self, single_data, **kwargs):
        return self.processor(single_data).input_values


class ToTensord(BaseTransform):
    def __init__(self, keys, **kwargs):
        super(ToTensord, self).__init__(keys, **kwargs)

    def _process(self, single_data, **kwargs):
        return torch.tensor(single_data)


def get_transforms(model_name):
    return Compose(
        [
            LoadAudio(keys=['audio']),
            ProcessAudio(keys=['audio'], model_name=model_name),
            ToTensord(keys=['label']),
        ]
    )
