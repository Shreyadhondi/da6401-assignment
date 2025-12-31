"""
Simple configurable CNN for DA6401 Assignment 2 (Part A, Q1 + Sweeps).

Architecture:
    Input (3 x H x W)
      -> [Conv2d -> BatchNorm(optional) -> Activation -> MaxPool2d] x 5
      -> Flatten
      -> Dense (hidden)
      -> Activation
      -> Dropout
      -> Output layer (num_classes)
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple
import torch
import torch.nn as nn


def get_activation(name: str) -> nn.Module:
    """Map string to activation function."""
    name = name.lower()
    if name == "relu":
        return nn.ReLU(inplace=True)
    if name == "gelu":
        return nn.GELU()
    if name == "silu":
        return nn.SiLU(inplace=True)
    if name == "mish":
        class Mish(nn.Module):
            def forward(self, x):
                return x * torch.tanh(nn.functional.softplus(x))
        return Mish()
    raise ValueError(f"Unsupported activation: {name}")


@dataclass
class CNNConfig:
    in_channels: int = 3
    num_classes: int = 10
    conv_channels: List[int] = None
    kernel_size: int = 3
    activation: str = "relu"
    dense_units: int = 256
    dropout: float = 0.5
    batch_norm: bool = False
    image_size: Tuple[int, int] = (224, 224)

    def __post_init__(self):
        if self.conv_channels is None:
            self.conv_channels = [32, 64, 128, 256, 256]


class SimpleCNN(nn.Module):
    def __init__(self, cfg: CNNConfig):
        super().__init__()
        self.cfg = cfg

        conv_blocks = []
        in_c = cfg.in_channels
        padding = cfg.kernel_size // 2

        # ---------------------------------------------------------
        # Build convolutional blocks
        # ---------------------------------------------------------
        for out_c in cfg.conv_channels:
            layers = [
                nn.Conv2d(
                    in_c, out_c,
                    kernel_size=cfg.kernel_size,
                    padding=padding,
                    bias=not cfg.batch_norm
                )
            ]
            if cfg.batch_norm:
                layers.append(nn.BatchNorm2d(out_c))

            layers.append(get_activation(cfg.activation))
            layers.append(nn.MaxPool2d(2, 2))

            conv_blocks.append(nn.Sequential(*layers))
            in_c = out_c

        self.conv_blocks = nn.Sequential(*conv_blocks)

        # ---------------------------------------------------------
        # Compute flattened dimension
        # ---------------------------------------------------------
        with torch.no_grad():
            dummy = torch.zeros(1, cfg.in_channels, *cfg.image_size)
            out = self.conv_blocks(dummy)
            self.flat_dim = out.flatten(1).shape[1]

        # ---------------------------------------------------------
        # Fully-connected layers
        # ---------------------------------------------------------
        self.fc_hidden = nn.Linear(self.flat_dim, cfg.dense_units)
        self.fc_act = get_activation(cfg.activation)
        self.dropout = nn.Dropout(cfg.dropout)
        self.fc_out = nn.Linear(cfg.dense_units, cfg.num_classes)

    # -------------------------------------------------------------
    def forward(self, x):
        x = self.conv_blocks(x)
        x = x.flatten(1)
        x = self.fc_hidden(x)
        x = self.fc_act(x)
        x = self.dropout(x)
        x = self.fc_out(x)
        return x

    def num_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
