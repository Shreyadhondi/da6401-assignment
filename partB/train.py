"""
Fine-tune a pretrained ResNet-50 on the iNaturalist dataset (Part B).

- Uses the same data splits as Part A: data/train, data/val, data/test
-Loads a pretrained ResNet-50 and replaces the final layer with 10 outputs.
- Applies the chosen fine-tuning strategy (from configs/partB_config.yaml).
- Trains for a fixed number of epochs (e.g. 15).
- Logs metrics to Weights & Biases (W&B) for record/plots.
- Saves the best model (by validation accuracy) to disk.
"""

import os
import copy
import random
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import wandb

from partB.model import PretrainedConfig, build_resnet50


# -------------------------------------------------------
# Utility: set random seeds (for reproducibility)
# -------------------------------------------------------
def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # make CuDNN deterministic
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# -------------------------------------------------------
# Utility: load YAML config
# -------------------------------------------------------
def load_config() -> dict:
    """
    Load configs/partB_config.yaml and return as a Python dict.
    """
    config_path = Path("configs") / "partB_config.yaml"
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg


# -------------------------------------------------------
# Data transforms and dataloaders
# -------------------------------------------------------
def build_transforms(cfg: dict, train: bool = True) -> transforms.Compose:
    """
    Build torchvision transforms for train/val/test.

    For train:
      - optional augmentations (flip, rotation, random crop)
      - resize/crop to the model's expected image size
      - convert to tensor and normalize

    For val/test:
      - just resize (center-crop is optional, we keep it simple)
      - convert to tensor and normalize
    """
    img_h, img_w = cfg["model"]["image_size"]
    data_cfg = cfg["data"]

    tfms = []

    if train:
        aug = data_cfg["augmentations"]

        if aug.get("horizontal_flip", False):
            tfms.append(transforms.RandomHorizontalFlip())

        deg = aug.get("random_rotation_degrees", 0)
        if deg > 0:
            tfms.append(transforms.RandomRotation(deg))

        scale = aug.get("random_crop_scale", [1.0, 1.0])
        tfms.append(
            transforms.RandomResizedCrop((img_h, img_w), scale=tuple(scale))
        )
    else:
        tfms.append(transforms.Resize((img_h, img_w)))

    # Convert to tensor and normalize with ImageNet mean/std
    tfms.append(transforms.ToTensor())
    tfms.append(
        transforms.Normalize(
            mean=data_cfg["mean"],
            std=data_cfg["std"],
        )
    )

    return transforms.Compose(tfms)


def build_dataloaders(cfg: dict) -> Tuple[DataLoader, DataLoader, DataLoader, int]:
    """
    Create DataLoaders for train, val and test splits.

    We assume the following folder structure:
      data/
        train/ Amphibia/ ... Reptilia/
        val/   Amphibia/ ... Reptilia/
        test/  Amphibia/ ... Reptilia/
    """
    train_tfms = build_transforms(cfg, train=True)
    val_tfms = build_transforms(cfg, train=False)
    test_tfms = build_transforms(cfg, train=False)

    data_cfg = cfg["data"]
    train_dir = data_cfg["train_dir"]
    val_dir = data_cfg["val_dir"]
    test_dir = data_cfg["test_dir"]

    train_set = datasets.ImageFolder(train_dir, transform=train_tfms)
    val_set = datasets.ImageFolder(val_dir, transform=val_tfms)
    test_set = datasets.ImageFolder(test_dir, transform=test_tfms)

    num_classes = len(train_set.classes)

    train_loader = DataLoader(
        train_set,
        batch_size=cfg["training"]["batch_size"],
        shuffle=True,
        num_workers=cfg["training"]["num_workers"],
        pin_memory=torch.cuda.is_available(),
    )

    val_loader = DataLoader(
        val_set,
        batch_size=cfg["training"]["batch_size"],
        shuffle=False,
        num_workers=cfg["training"]["num_workers"],
        pin_memory=torch.cuda.is_available(),
    )

    test_loader = DataLoader(
        test_set,
        batch_size=cfg["training"]["batch_size"],
        shuffle=False,
        num_workers=cfg["training"]["num_workers"],
        pin_memory=torch.cuda.is_available(),
    )

    return train_loader, val_loader, test_loader, num_classes


# -------------------------------------------------------
# Training / evaluation helpers
# -------------------------------------------------------
def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
) -> Tuple[float, float]:
    """
    Train for a single epoch and return (loss, accuracy).
    """
    model.train()
    running_loss = 0.0
    running_correct = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1)
        running_correct += (preds == labels).sum().item()

    epoch_loss = running_loss / len(loader.dataset)
    epoch_acc = running_correct / len(loader.dataset)
    return epoch_loss, epoch_acc


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    """
    Evaluate the model (no gradient) and return (loss, accuracy).
    """
    model.eval()
    running_loss = 0.0
    running_correct = 0

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1)
            running_correct += (preds == labels).sum().item()

    epoch_loss = running_loss / len(loader.dataset)
    epoch_acc = running_correct / len(loader.dataset)
    return epoch_loss, epoch_acc


# -------------------------------------------------------
# Main training script
# -------------------------------------------------------
def main() -> None:
    set_seed(42)

    cfg = load_config()
    train_cfg = cfg["training"]

    # Choose device
    device_str = train_cfg.get("device", "cuda")
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Build data loaders
    train_loader, val_loader, test_loader, num_classes = build_dataloaders(cfg)

    # Build model config and model
    model_cfg = PretrainedConfig(
        num_classes=num_classes,
        finetune_strategy=train_cfg["finetune_strategy"],
    )
    model = build_resnet50(model_cfg).to(device)
    print("Model built with fine-tuning strategy:", model_cfg.finetune_strategy)

    # Count trainable parameters (for curiosity / report)
    trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    print(f"Trainable parameters: {trainable_params:,}")

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=train_cfg["learning_rate"],
        weight_decay=train_cfg["weight_decay"],
    )

    # ---------------------------------------------
    # W&B Init (simple logging, no sweep here)
    # ---------------------------------------------
    run = wandb.init(
    project="da24m019-assignment2",   # SAME project as Part-A
    group="partB",                   # groups Part-B runs together
    job_type="finetune",             # semantic label
    tags=["partB", "resnet50"],       # easy filtering in UI
    config={
        "model_name": cfg["model"]["name"],
        "finetune_strategy": model_cfg.finetune_strategy,
        "batch_size": train_cfg["batch_size"],
        "num_epochs": train_cfg["num_epochs"],
        "learning_rate": train_cfg["learning_rate"],
        "weight_decay": train_cfg["weight_decay"],
    },
)

    # Give the run a human-readable name
    cfg_w = wandb.config
    run.name = (
        f"{cfg_w.model_name}_"
        f"{cfg_w.finetune_strategy}_"
        f"lr-{cfg_w.learning_rate}_"
        f"bs-{cfg_w.batch_size}"
    )

    # Watch model gradients/weights (optional but nice)
    wandb.watch(model, log="all", log_freq=50)

    num_epochs = train_cfg["num_epochs"]

    best_val_acc = 0.0
    best_state_dict = copy.deepcopy(model.state_dict())

    # ---------------------------------------------
    # Training loop
    # ---------------------------------------------
    for epoch in range(1, num_epochs + 1):
        print(f"\nEpoch {epoch}/{num_epochs}")

        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_acc = evaluate(
            model, val_loader, criterion, device
        )

        print(
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}\n"
            f"Val   Loss: {val_loss:.4f} | Val   Acc: {val_acc:.4f}"
        )

        # Log to W&B
        wandb.log(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
            }
        )

        # Keep the best model (by validation accuracy)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state_dict = copy.deepcopy(model.state_dict())

    print(f"\nBest validation accuracy: {best_val_acc:.4f}")

    # ---------------------------------------------
    # Evaluate best model on test set
    # ---------------------------------------------
    model.load_state_dict(best_state_dict)
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)

    print("\n==== Test Performance (Best Fine-tuned Model) ====")
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Acc : {test_acc:.4f}")

    wandb.log({"test_loss": test_loss, "test_acc": test_acc})

    # ---------------------------------------------
    # Save best model to disk
    # ---------------------------------------------
    save_dir = Path("partB")
    save_dir.mkdir(exist_ok=True)
    model_path = save_dir / "best_resnet50_partB.pth"
    torch.save(best_state_dict, model_path)
    print(f"Saved best model weights to: {model_path}")

    run.finish()


if __name__ == "__main__":
    main()
