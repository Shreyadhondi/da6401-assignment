"""
Evaluate best CNN configuration on iNaturalist test set.

Steps:
1. Define BEST_CONFIG from W&B sweep results.
2. Train SimpleCNN with these hyperparameters on the training set
   (using the same data pipeline as in train.py).
3. Evaluate accuracy on the test set.
4. Save a 10x3 grid of test images with predicted + true labels as
   partA/best_model_test_grid.png
"""

import random
from typing import Tuple, List

import yaml
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

from partA.model import SimpleCNN, CNNConfig


# -----------------------------#
# 1. Best config from sweep
# -----------------------------#
BEST_CONFIG = {
    "activation": "gelu",
    "conv_channels": [32, 64, 128, 128, 256],
    "optimizer": "adam",
    "learning_rate": 0.001,
    "weight_decay": 0.0001,
    "batch_size": 32,
    "dropout": 0.4,
    "use_batchnorm": False,
    "use_augmentations": False,
    "num_epochs": 10,
    "kernel_size": 3,
    "dense_units": 256,
}


# -----------------------------#
# 2. Config + transforms
# -----------------------------#
def load_yaml_config():
    """Load base YAML config (paths, image size, mean/std, etc.)."""
    with open("configs/partA_config.yaml", "r") as f:
        return yaml.safe_load(f)


def build_transforms(cfg, train: bool, use_augmentations: bool) -> transforms.Compose:
    img_h, img_w = cfg["model"]["image_size"]
    tfms: List[transforms.Transform] = []

    if train and use_augmentations:
        aug = cfg["data"]["augmentations"]

        if aug.get("horizontal_flip", False):
            tfms.append(transforms.RandomHorizontalFlip())

        rot = aug.get("random_rotation_degrees", 0)
        if rot > 0:
            tfms.append(transforms.RandomRotation(rot))

        scale = aug.get("random_crop_scale", [1.0, 1.0])
        tfms.append(transforms.RandomResizedCrop((img_h, img_w), scale=tuple(scale)))
    else:
        tfms.append(transforms.Resize((img_h, img_w)))

    tfms.append(transforms.ToTensor())
    tfms.append(
        transforms.Normalize(
            mean=cfg["data"]["mean"],
            std=cfg["data"]["std"],
        )
    )
    return transforms.Compose(tfms)


def build_train_full_loader(
    cfg, batch_size: int, num_workers: int, use_augmentations: bool
) -> Tuple[DataLoader, int]:
    """
    Build a DataLoader for full training data.

    We concatenate train + val folders so that we train on all labeled data.
    (We still haven't touched test during training.)
    """
    train_tfms = build_transforms(cfg, train=True, use_augmentations=use_augmentations)

    train_dir = cfg["data"]["train_dir"]
    val_dir = cfg["data"]["val_dir"]

    train_set = datasets.ImageFolder(train_dir, transform=train_tfms)
    val_set = datasets.ImageFolder(val_dir, transform=train_tfms)

    full_train = ConcatDataset([train_set, val_set])

    train_loader = DataLoader(
        full_train,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    # Class names come from any of the underlying datasets
    num_classes = len(train_set.classes)
    return train_loader, num_classes


def build_test_loader(cfg, batch_size: int) -> Tuple[DataLoader, datasets.ImageFolder]:
    """Build DataLoader and underlying Dataset for the test split."""
    test_tfms = build_transforms(cfg, train=False, use_augmentations=False)
    test_dir = cfg["data"]["test_dir"]
    test_set = datasets.ImageFolder(test_dir, transform=test_tfms)

    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=cfg["training"]["num_workers"],
        pin_memory=torch.cuda.is_available(),
    )
    return test_loader, test_set


# -----------------------------#
# 3. Train / Eval routines
# -----------------------------#
def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
) -> Tuple[float, float]:
    model.train()
    total_loss = 0.0
    total_correct = 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * x.size(0)
        total_correct += (out.argmax(1) == y).sum().item()

    loss = total_loss / len(loader.dataset)
    acc = total_correct / len(loader.dataset)
    return loss, acc


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_correct = 0

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)

            out = model(x)
            loss = criterion(out, y)

            total_loss += loss.item() * x.size(0)
            total_correct += (out.argmax(1) == y).sum().item()

    loss = total_loss / len(loader.dataset)
    acc = total_correct / len(loader.dataset)
    return loss, acc


# -----------------------------#
# 4. 10x3 prediction grid
# -----------------------------#
def save_prediction_grid_10x3(
    cfg,
    model: nn.Module,
    dataset: datasets.ImageFolder,
    device: torch.device,
    out_path: str = "partA/best_model_test_grid.png",
) -> None:
    """
    Save a 10x3 grid (30 images) with predicted + true labels.
    """
    model.eval()
    num_images = 30
    indices = random.sample(range(len(dataset)), num_images)

    class_names = dataset.classes
    mean = np.array(cfg["data"]["mean"])
    std = np.array(cfg["data"]["std"])

    fig, axes = plt.subplots(10, 3, figsize=(12, 24))

    for ax, idx in zip(axes.flatten(), indices):
        img_tensor, true_label = dataset[idx]
        img = img_tensor.unsqueeze(0).to(device)

        with torch.no_grad():
            out = model(img)
            pred_label = out.argmax(1).item()

        # Unnormalize for display: (x * std + mean)
        img_np = img_tensor.permute(1, 2, 0).cpu().numpy()
        img_np = (img_np * std) + mean
        img_np = np.clip(img_np, 0.0, 1.0)

        ax.imshow(img_np)
        ax.axis("off")
        ax.set_title(
            f"T: {class_names[true_label]}\nP: {class_names[pred_label]}",
            fontsize=8,
        )

    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"Saved 10x3 prediction grid to: {out_path}")


# -----------------------------#
# 5. Main
# -----------------------------#
def main():
    cfg_yaml = load_yaml_config()
    train_yaml = cfg_yaml["training"]
    model_yaml = cfg_yaml["model"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Build loaders
    train_loader, num_classes = build_train_full_loader(
        cfg_yaml,
        batch_size=BEST_CONFIG["batch_size"],
        num_workers=train_yaml["num_workers"],
        use_augmentations=BEST_CONFIG["use_augmentations"],
    )

    test_loader, test_set = build_test_loader(
        cfg_yaml,
        batch_size=BEST_CONFIG["batch_size"],
    )

    # Build model
    model_cfg = CNNConfig(
        num_classes=num_classes,
        in_channels=3,
        conv_channels=list(BEST_CONFIG["conv_channels"]),
        kernel_size=BEST_CONFIG["kernel_size"],
        activation=BEST_CONFIG["activation"],
        dense_units=BEST_CONFIG["dense_units"],
        dropout=BEST_CONFIG["dropout"],
        batch_norm=BEST_CONFIG["use_batchnorm"],
        image_size=tuple(model_yaml["image_size"]),
    )

    model = SimpleCNN(model_cfg).to(device)
    print("Model parameters:", model.num_parameters())

    # Optimizer
    if BEST_CONFIG["optimizer"] == "adam":
        optimizer = optim.Adam(
            model.parameters(),
            lr=BEST_CONFIG["learning_rate"],
            weight_decay=BEST_CONFIG["weight_decay"],
        )
    else:
        optimizer = optim.SGD(
            model.parameters(),
            lr=BEST_CONFIG["learning_rate"],
            momentum=0.9,
            weight_decay=BEST_CONFIG["weight_decay"],
        )

    criterion = nn.CrossEntropyLoss()

    # -------- Training on full train data (train+val) --------
    for epoch in range(BEST_CONFIG["num_epochs"]):
        print(f"\nEpoch {epoch + 1}/{BEST_CONFIG['num_epochs']}")
        tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        print(f"Train Loss: {tr_loss:.4f} | Train Acc: {tr_acc:.4f}")

    # -------- Final evaluation on test set --------
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    print("\n==== Test Performance (Best Config) ====")
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Acc : {test_acc:.4f}")

    # Save 10x3 grid with predictions
    save_prediction_grid_10x3(cfg_yaml, model, test_set, device)


if __name__ == "__main__":
    main()
