#!/usr/bin/env python
# coding: utf-8

import os
import pickle
import subprocess
import torch
import torch.nn as nn
from torchvision import models

from download_and_prepare_models import LoadedResNet

MODELS = [
    '/home/ubuntu/models/moco/mimic/r50/moco-v3_r50_100e_mimic_after_imagenet_deit.pth'
]

if __name__ == '__main__':
    for model_path in MODELS:
        model = LoadedResNet("moco-v3", model_path=model_path)
        state_dict = model.state_dict()
        state_dict = {key.replace('model.', ''): val for key, val in state_dict.items()}
        save_path = f'{model_path.split(".pth")[0]}_prepped.pth'
        print(f'Saving to {save_path}')
        torch.save(state_dict, save_path)
