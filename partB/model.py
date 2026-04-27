"""
partB/model.py

Pretrained ResNet-50 wrapper for DA6401 Assignment 2 – Part B.

- Loads ResNet-50 with ImageNet weights from torchvision.
- Replaces the final classification layer to predict 10 iNaturalist classes.
- Applies a chosen fine-tuning strategy by freezing/unfreezing layers.

We define three strategies (for your report):

1. "only_fc":
   - Freeze all backbone layers.
   - Train only the final fully-connected (fc) layer.
2. "layer4_and_fc"  (THIS is what we will actually use):
   - Freeze conv1 + layer1 + layer2 + layer3.
   - Train layer4 and the final fc layer.
3. "full_finetune":
   - Train all layers (no freezing).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights


FineTuneStrategy = Literal["only_fc", "layer4_and_fc", "full_finetune"]


@dataclass
class PretrainedConfig:
    """Configuration for the pretrained model."""

    num_classes: int = 10
    finetune_strategy: FineTuneStrategy = "layer4_and_fc"


def build_resnet50(cfg: PretrainedConfig) -> nn.Module:
    """
    Build a ResNet-50 model pre-trained on ImageNet and adapt it for 10 classes.

    Steps:
    1. Load torchvision's ResNet-50 with pretrained ImageNet weights.
    2. Replace the final 'fc' layer with a new Linear layer with 10 outputs.
    3. Freeze or unfreeze layers according to cfg.finetune_strategy.
    """
    # 1) Load pretrained weights
    weights = ResNet50_Weights.IMAGENET1K_V1
    model = resnet50(weights=weights)

    # 2) Replace the last fully-connected layer:
    #    original: Linear(in_features=2048, out_features=1000)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, cfg.num_classes)

    # 3) Apply fine-tuning strategy
    strategy = cfg.finetune_strategy

    if strategy == "only_fc":
        # Freeze ALL parameters
        for param in model.parameters():
            param.requires_grad = False
        # Unfreeze ONLY the final classification head
        for param in model.fc.parameters():
            param.requires_grad = True

    elif strategy == "layer4_and_fc":
        # First freeze everything
        for param in model.parameters():
            param.requires_grad = False

        # Then unfreeze layer4 and fc layer
        for name, param in model.named_parameters():
            # layer4.* are the deepest residual blocks
            if name.startswith("layer4.") or name.startswith("fc."):
                param.requires_grad = True

    elif strategy == "full_finetune":
        # No freezing at all: train the entire network
        for param in model.parameters():
            param.requires_grad = True
    else:
        raise ValueError(f"Unknown finetune_strategy: {strategy}")

    return model
