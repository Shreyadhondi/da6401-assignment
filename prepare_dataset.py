"""
Why are we doing this?

In the DA6401 – Assignment 2 description, the iNaturalist subset is
described as having two folders: `train/` and `test/`. However, the
downloaded `nature_12K` dataset actually contains `train/` and `val/`
folders.

To match the assignment instructions, I have made the following choices:

1. We treat the existing `val/` folder from `nature_12K` as the **test set**.
2. From the existing `train/` folder, we create our own **validation set** by
   taking 20% of the images from each class (stratified split).
3. The remaining 80% of images in each class are used as the **new train set**.

After running this script, we will have:

    data/
        train/    <- 80% of original train images (per class)
        val/      <- 20% of original train images (per class)
        test/     <- all images from original val/ (per class)

This structure will then be used in our Part A and Part B experiments.
"""

import random
import shutil
from pathlib import Path

VAL_RATIO = 0.2  # 20% of train data per class for validation
RANDOM_SEED = 42


def copy_tree(src: Path, dst: Path) -> None:
    """
    Recursively copy directory structure from src to dst.

    Used to copy original `val/` (from nature_12K) into our new `test/` folder.
    """
    for class_dir in src.iterdir():
        if not class_dir.is_dir():
            continue

        target_class_dir = dst / class_dir.name
        target_class_dir.mkdir(parents=True, exist_ok=True)

        for img_path in class_dir.iterdir():
            if img_path.is_file():
                shutil.copy2(img_path, target_class_dir / img_path.name)


def split_train_val(orig_train: Path, new_train: Path, new_val: Path) -> None:
    """
    For each class in orig_train, split images into 80% train / 20% val.

    The split is done per class, with a fixed random seed for reproducibility.
    Files are MOVED into `new_train` and `new_val`.
    """
    random.seed(RANDOM_SEED)

    for class_dir in orig_train.iterdir():
        if not class_dir.is_dir():
            continue

        images = [p for p in class_dir.iterdir() if p.is_file()]
        if not images:
            print(f"[WARN] No images found in class: {class_dir.name}")
            continue

        # Reproducible shuffle
        images.sort()  # deterministic order before shuffle
        random.shuffle(images)

        n_total = len(images)
        n_val = max(1, int(round(VAL_RATIO * n_total)))
        val_images = images[:n_val]
        train_images = images[n_val:]

        print(
            f"[INFO] Class '{class_dir.name}': "
            f"{n_total} total -> {len(train_images)} train, {len(val_images)} val"
        )

        # Ensure target dirs exist
        train_class_dir = new_train / class_dir.name
        val_class_dir = new_val / class_dir.name
        train_class_dir.mkdir(parents=True, exist_ok=True)
        val_class_dir.mkdir(parents=True, exist_ok=True)

        # Move files
        for img_path in train_images:
            shutil.move(str(img_path), train_class_dir / img_path.name)

        for img_path in val_images:
            shutil.move(str(img_path), val_class_dir / img_path.name)


def main() -> None:
    root = Path(__file__).resolve().parent

    # Original dataset location (what you already have)
    nature_root = root / "nature_12K" / "inaturalist_12K"
    orig_train = nature_root / "train"
    orig_val = nature_root / "val"

    if not orig_train.exists() or not orig_val.exists():
        raise FileNotFoundError(
            "Expected 'nature_12K/train' and 'nature_12K/val' folders "
            "relative to this script. Please check your dataset location."
        )

    # New target structure
    data_root = root / "data"
    new_train = data_root / "train"
    new_val = data_root / "val"
    new_test = data_root / "test"

    if data_root.exists():
        raise RuntimeError(
            f"'data/' folder already exists at {data_root}. "
            "To avoid messing up existing splits, please delete or rename "
            "it before running this script again."
        )

    # Create root data directory
    data_root.mkdir(parents=True, exist_ok=False)

    print("[STEP] Copying original 'val/' to new 'test/' ...")
    copy_tree(orig_val, new_test)

    print("[STEP] Splitting original 'train/' into new 'train/' and 'val/' ...")
    split_train_val(orig_train, new_train, new_val)

    print("\n[DONE] Dataset prepared.")
    print(f"New structure created under: {data_root}")
    print("You can now use 'data/train', 'data/val', and 'data/test' "
          "in your training scripts.")


if __name__ == "__main__":
    main()
