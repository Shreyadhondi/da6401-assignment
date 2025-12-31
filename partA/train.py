import os
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import wandb

# ------------------------------------------------------------------
# Robust import: works for both
#   - python -m partA.train  (root as package)
#   - python partA/train.py  (script inside folder)
# ------------------------------------------------------------------
try:
    from partA.model import SimpleCNN, CNNConfig  # when run as module
except ImportError:
    from model import SimpleCNN, CNNConfig        # when run as script


# -------------------------------------------------------
# Load YAML config
# -------------------------------------------------------
def load_yaml_config():
    with open("configs/partA_config.yaml", "r") as f:
        return yaml.safe_load(f)


# -------------------------------------------------------
# Data transforms
# -------------------------------------------------------
def build_transforms(cfg, train: bool = True, use_augmentations: bool = True):
    img_h, img_w = cfg["model"]["image_size"]
    tfms = []

    if train and use_augmentations:
        aug = cfg["data"]["augmentations"]

        if aug.get("horizontal_flip", False):
            tfms.append(transforms.RandomHorizontalFlip())

        rot = aug.get("random_rotation_degrees", 0)
        if rot > 0:
            tfms.append(transforms.RandomRotation(rot))

        scale = aug.get("random_crop_scale", [1.0, 1.0])
        tfms.append(
            transforms.RandomResizedCrop((img_h, img_w), scale=tuple(scale))
        )
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


# -------------------------------------------------------
# Build dataloaders
# -------------------------------------------------------
def build_dataloaders(cfg, batch_size, num_workers, use_augmentations: bool = True):
    train_tfms = build_transforms(cfg, train=True, use_augmentations=use_augmentations)
    val_tfms = build_transforms(cfg, train=False, use_augmentations=False)

    train_set = datasets.ImageFolder(cfg["data"]["train_dir"], transform=train_tfms)
    val_set = datasets.ImageFolder(cfg["data"]["val_dir"], transform=val_tfms)

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    return train_loader, val_loader, len(train_set.classes)


# -------------------------------------------------------
# Train one epoch
# -------------------------------------------------------
def train_one_epoch(model, loader, criterion, optimizer, device):
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

    return total_loss / len(loader.dataset), total_correct / len(loader.dataset)


# -------------------------------------------------------
# Validation
# -------------------------------------------------------
def evaluate(model, loader, criterion, device):
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

    return total_loss / len(loader.dataset), total_correct / len(loader.dataset)


# -------------------------------------------------------
# Main
# -------------------------------------------------------
def main():
    base_cfg = load_yaml_config()
    model_yaml = base_cfg["model"]
    train_yaml = base_cfg["training"]

    # ---------------------------------------------
    # W&B init - sweep overrides happen here
    # ---------------------------------------------
    run = wandb.init(
        project="da6401-assignment-partA",   # match sweep project
        config={
            "batch_size": train_yaml["batch_size"],
            "num_epochs": train_yaml["num_epochs"],
            "learning_rate": train_yaml["learning_rate"],
            "weight_decay": train_yaml["weight_decay"],
            "conv_channels": model_yaml["conv_channels"],
            "kernel_size": model_yaml["kernel_size"],
            "activation": model_yaml["activation"],
            "dense_units": model_yaml["dense_units"],
            "dropout": model_yaml["dropout"],
            "optimizer": train_yaml["optimizer"],
            # default names (these are what *we* use)
            "batch_norm": False,
            "augmentations": True,
        },
    )

    cfg = wandb.config

    # Also support sweeps that might be using different key names
    batch_norm = getattr(cfg, "batch_norm", getattr(cfg, "use_batchnorm", False))
    augmentations = getattr(
        cfg, "augmentations", getattr(cfg, "use_augmentations", True)
    )

    # Meaningful dynamic run name
    run.name = (
        f"act-{cfg.activation}_"
        f"conv-{cfg.conv_channels}_"
        f"opt-{cfg.optimizer}_"
        f"bn-{batch_norm}_"
        f"aug-{augmentations}_"
        f"bs-{cfg.batch_size}"
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Load data
    train_loader, val_loader, num_classes = build_dataloaders(
        base_cfg,
        batch_size=cfg.batch_size,
        num_workers=train_yaml["num_workers"],
        use_augmentations=augmentations,
    )

    # Build model
    model_cfg = CNNConfig(
        num_classes=num_classes,
        in_channels=3,
        conv_channels=list(cfg.conv_channels),
        kernel_size=cfg.kernel_size,
        activation=cfg.activation,
        dense_units=cfg.dense_units,
        dropout=cfg.dropout,
        batch_norm=batch_norm,
        image_size=tuple(model_yaml["image_size"]),
    )

    model = SimpleCNN(model_cfg).to(device)
    print("Model parameters:", model.num_parameters())

    wandb.watch(model)

    # Optimizer
    if cfg.optimizer == "adam":
        optimizer = optim.Adam(
            model.parameters(),
            lr=cfg.learning_rate,
            weight_decay=cfg.weight_decay,
        )
    else:
        optimizer = optim.SGD(
            model.parameters(),
            lr=cfg.learning_rate,
            momentum=0.9,
            weight_decay=cfg.weight_decay,
        )

    criterion = nn.CrossEntropyLoss()

    # Training
    num_epochs = getattr(cfg, "num_epochs", train_yaml["num_epochs"])

    for ep in range(num_epochs):
        print(f"\nEpoch {ep + 1}/{num_epochs}")

        tr_loss, tr_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        va_loss, va_acc = evaluate(model, val_loader, criterion, device)

        print(f"Train Loss: {tr_loss:.4f} | Train Acc: {tr_acc:.4f}")
        print(f"Val   Loss: {va_loss:.4f} | Val   Acc: {va_acc:.4f}")

        wandb.log(
            {
                "epoch": ep + 1,
                "train_loss": tr_loss,
                "train_acc": tr_acc,
                "val_loss": va_loss,
                "val_acc": va_acc,
            }
        )

    run.finish()


if __name__ == "__main__":
    main()
