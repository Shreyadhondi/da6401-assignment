"""
Simple configurable CNN for DA6401 Assignment 2 (Part A, Q1).

Architecture:
    Input (3 x H x W)
      -> [Conv2d -> Activation -> MaxPool2d] x 5
      -> Flatten
      -> Dense (hidden)
      -> Activation
      -> Dropout
      -> Output layer (num_classes)

All important parts are configurable through arguments, which we will
later read from configs/partA_config.yaml.

This file focuses only on defining the model, not training.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import torch
import torch.nn as nn


def get_activation(name: str) -> nn.Module:
    """
    Return an activation module given its name.

    Supported: "relu", "gelu", "silu", "mish".
    """
    name = name.lower()
    if name == "relu":
        return nn.ReLU(inplace=True)
    if name == "gelu":
        return nn.GELU()
    if name == "silu":
        return nn.SiLU(inplace=True)
    if name == "mish":
        # PyTorch doesn't have Mish built-in in all versions; simple implementation
        class Mish(nn.Module):
            def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
                return x * torch.tanh(nn.functional.softplus(x))

        return Mish()

    raise ValueError(f"Unsupported activation: {name}")


@dataclass
class CNNConfig:
    """
    Small dataclass to keep model hyperparameters together.

    These fields will correspond to entries in configs/partA_config.yaml.
    """
    in_channels: int = 3
    num_classes: int = 10
    conv_channels: List[int] = None  # will set default in __post_init__
    kernel_size: int = 3
    activation: str = "relu"
    dense_units: int = 256
    dropout: float = 0.5
    image_size: Tuple[int, int] = (224, 224)  # (H, W)

    def __post_init__(self) -> None:
        if self.conv_channels is None:
            self.conv_channels = [32, 64, 128, 256, 256]


class SimpleCNN(nn.Module):
    """
    CNN model used in Part A, Question 1.

    - 5 conv blocks (Conv -> Activation -> MaxPool)
    - 1 fully connected hidden layer
    - 1 output layer with `num_classes` units
    """

    def __init__(self, cfg: CNNConfig) -> None:
        super().__init__()
        self.cfg = cfg

        activation_layer = get_activation(cfg.activation)

        # ----- Convolutional blocks -----
        conv_layers: List[nn.Module] = []

        in_channels = cfg.in_channels
        padding = cfg.kernel_size // 2  # to roughly preserve HxW before pooling

        for out_channels in cfg.conv_channels:
            block = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=cfg.kernel_size,
                    padding=padding,
                    bias=True,
                ),
                get_activation(cfg.activation),  # new instance each time
                nn.MaxPool2d(kernel_size=2, stride=2),
            )
            conv_layers.append(block)
            in_channels = out_channels

        self.conv_blocks = nn.Sequential(*conv_layers)

        # ----- Infer flattened size after conv + pooling -----
        with torch.no_grad():
            h, w = cfg.image_size
            dummy = torch.zeros(1, cfg.in_channels, h, w)
            out = self.conv_blocks(dummy)
            self._flatten_dim = out.view(1, -1).shape[1]

        # ----- Fully connected part -----
        self.fc_hidden = nn.Linear(self._flatten_dim, cfg.dense_units)
        self.fc_activation = activation_layer
        self.dropout = nn.Dropout(p=cfg.dropout)
        self.fc_out = nn.Linear(cfg.dense_units, cfg.num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        """
        Forward pass through the network.
        """
        x = self.conv_blocks(x)          # [B, C, H', W']
        x = torch.flatten(x, 1)          # [B, C * H' * W']
        x = self.fc_hidden(x)            # [B, dense_units]
        x = self.fc_activation(x)
        x = self.dropout(x)
        x = self.fc_out(x)               # [B, num_classes]
        return x

    def num_parameters(self) -> int:
        """
        Utility function: return total number of trainable parameters.

        This is for sanity-checking; for the written part of Q1 you will
        still derive the formula using m, k, n.
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
