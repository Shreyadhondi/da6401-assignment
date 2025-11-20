import os
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from partA.model import SimpleCNN, CNNConfig


def load_config():
    """Loads YAML config file."""
    config_path = os.path.join("configs", "partA_config.yaml")
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg


def build_transforms(cfg, train=True):
    """Builds transforms using config."""
    img_h, img_w = cfg["model"]["image_size"]

    transform_list = []

    if train and cfg["data"]["augmentations"]["horizontal_flip"]:
        transform_list.append(transforms.RandomHorizontalFlip())

    if train and cfg["data"]["augmentations"]["random_rotation_degrees"] > 0:
        degrees = cfg["data"]["augmentations"]["random_rotation_degrees"]
        transform_list.append(transforms.RandomRotation(degrees))

    if train:
        scale_range = cfg["data"]["augmentations"]["random_crop_scale"]
        transform_list.append(transforms.RandomResizedCrop((img_h, img_w),
                                                           scale=tuple(scale_range)))
    else:
        transform_list.append(transforms.Resize((img_h, img_w)))

    # Normalization
    transform_list.append(transforms.ToTensor())
    transform_list.append(transforms.Normalize(mean=cfg["data"]["mean"],
                                               std=cfg["data"]["std"]))

    return transforms.Compose(transform_list)


def build_dataloaders(cfg):
    """Creates train & val dataloaders."""
    train_dir = cfg["data"]["train_dir"]
    val_dir = cfg["data"]["val_dir"]
    batch_size = cfg["training"]["batch_size"]
    num_workers = cfg["training"]["num_workers"]

    train_tfms = build_transforms(cfg, train=True)
    val_tfms = build_transforms(cfg, train=False)

    train_set = datasets.ImageFolder(train_dir, transform=train_tfms)
    val_set = datasets.ImageFolder(val_dir, transform=val_tfms)

    train_loader = DataLoader(train_set, batch_size=batch_size,
                              shuffle=True, num_workers=num_workers)

    val_loader = DataLoader(val_set, batch_size=batch_size,
                            shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, len(train_set.classes)


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss, running_correct = 0.0, 0

    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * imgs.size(0)
        _, preds = outputs.max(1)
        running_correct += (preds == labels).sum().item()

    epoch_loss = running_loss / len(loader.dataset)
    epoch_acc = running_correct / len(loader.dataset)
    return epoch_loss, epoch_acc


def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss, running_correct = 0.0, 0

    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)

            outputs = model(imgs)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * imgs.size(0)
            _, preds = outputs.max(1)
            running_correct += (preds == labels).sum().item()

    epoch_loss = running_loss / len(loader.dataset)
    epoch_acc = running_correct / len(loader.dataset)
    return epoch_loss, epoch_acc


def main():
    cfg = load_config()
    device = torch.device(cfg["training"]["device"])

    print("Using device:", device)

    # Load dataloaders
    train_loader, val_loader, num_classes = build_dataloaders(cfg)

    # Build model config
    model_cfg = CNNConfig(
        num_classes=num_classes,
        in_channels=3,
        conv_channels=cfg["model"]["conv_channels"],
        kernel_size=cfg["model"]["kernel_size"],
        activation=cfg["model"]["activation"],
        dense_units=cfg["model"]["dense_units"],
        dropout=cfg["model"]["dropout"],
        image_size=cfg["model"]["image_size"]
    )

    model = SimpleCNN(model_cfg).to(device)
    print(f"Model parameters: {model.num_parameters()}")

    # Optimizer, Loss
    lr = cfg["training"]["learning_rate"]
    weight_decay = cfg["training"]["weight_decay"]

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()

    num_epochs = cfg["training"]["num_epochs"]

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")

        train_loss, train_acc = train_one_epoch(model, train_loader, criterion,
                                                optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"Val   Loss: {val_loss:.4f} | Val   Acc: {val_acc:.4f}")


if __name__ == "__main__":
    main()
