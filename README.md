# DA6401 Assignment 2

**Shreya Dhondi (DA24M019)**
Indian Institute of Technology Madras

---

## 📌 Overview

This repository contains the implementation for **Assignment 2** of the course **DA6401: Deep Learning**.

The assignment is divided into two main parts:

* **Part A**: Training a Convolutional Neural Network (CNN) from scratch on the iNaturalist dataset
* **Part B**: Fine-tuning a pretrained ResNet-50 model

All experiments are tracked using **Weights & Biases (W&B)** for reproducibility and analysis.

---

## 📁 Repository Structure

```
da6401-assignment/
│
├── configs/
│   ├── partA_config.yaml       # CNN training config
│   ├── partB_config.yaml       # ResNet50 fine-tuning config
│   └── sweep_config.yaml       # W&B sweep config
│
├── partA/
│   ├── model.py                # CNN architecture
│   ├── train.py                # Training + W&B logging
│   ├── evaluate_best.py        # Best model evaluation
│
├── partB/
│   ├── model.py                # ResNet50 wrapper
│   ├── train.py                # Fine-tuning script
│
├── prepare_dataset.py          # Dataset preprocessing
├── requirements.txt
└── README.md
```

---

## 📊 W&B Report

All experiments, sweeps, and results:

🔗 https://wandb.ai/shreyadhondi-indian-institute-of-technology-madras/da24m019-assignment2

---

## 📦 Dataset Preparation

### ⚠️ Important Note

The assignment description assumes the dataset contains:

```
train/
test/
```

However, the actual **iNaturalist 12K dataset** contains:

```
train/
val/
```

---

## 🔧 What I Did (Design Decision)

To strictly follow assignment requirements, I performed:

1. Treated original `val/` as **test set**
2. Split original `train/` into:

   * **80% → train**
   * **20% → validation**
3. Ensured **class-wise (stratified) split**

---

## 📁 Final Dataset Structure

After running preprocessing:

```
data/
├── train/   (80% of original train)
├── val/     (20% of original train)
└── test/    (original val)
```

---

## ⚙️ Setup Instructions

### 1. Clone Repository

```bash
git clone https://github.com/Shreyadhondi/da6401-assignment.git
cd da6401-assignment
```

---

### 2. Create Environment

```bash
conda create -n da6401 python=3.11
conda activate da6401
pip install -r requirements.txt
```

---

### 3. Download Dataset

Download **iNaturalist 12K dataset** and place it as:

```
nature_12K/inaturalist_12K/
    ├── train/
    └── val/
```

---

### 4. Prepare Dataset

```bash
python prepare_dataset.py
```

This creates the required structure in `data/`.

---

## 🧠 Part A: CNN from Scratch

### Architecture

* 5 Conv blocks:

  ```
  Conv → (BatchNorm optional) → Activation → MaxPool
  ```
* Flatten → Dense → Dropout → Output

Fully configurable via YAML.

---

### Train Model

```bash
python -m partA.train
```

---

### Hyperparameter Sweep (W&B)

```bash
wandb sweep configs/sweep_config.yaml
wandb agent <SWEEP_ID>
```

---

### Evaluate Best Model

```bash
python -m partA.evaluate_best
```

Outputs:

* Test accuracy
* 10×3 prediction grid image

---

## 🚀 Part B: Fine-Tuning ResNet-50

### Strategy Used

* Pretrained on ImageNet
* Fine-tuning strategy: `layer4_and_fc`

---

### Train Model

```bash
python -m partB.train
```

Outputs:

* Best validation model
* Test accuracy (~78%)
* Saved weights: `partB/best_resnet50_partB.pth`

---

## 📈 Results Summary

| Model           | Description              | Test Accuracy |
| --------------- | ------------------------ | ------------- |
| Part A CNN      | Best sweep configuration | ~36%          |
| Part B ResNet50 | Fine-tuned model         | ~78%          |

---

## 📦 Dependencies

All required packages:

```
requirements.txt
```

---

## 🔬 Weights & Biases (W&B)

This project uses W&B for experiment tracking.

### Behavior:

* Logs to **currently logged-in account**
* Project name: `da24m019-assignment2`

### Optional:

Disable logging:

```bash
wandb offline
```

Login to your account:

```bash
wandb login
```

---

## 📌 Reproducibility Notes

* Fixed random seed used in dataset split
* YAML configs control all hyperparameters
* Code supports both:

  ```
  python -m partA.train
  python partA/train.py
  ```

---

## 🔗 Important Links

* 📁 **GitHub Repository**:
  https://github.com/Shreyadhondi/da6401-assignment

* 📊 **W&B Project Dashboard**:
  https://wandb.ai/shreyadhondi-indian-institute-of-technology-madras/da24m019-assignment2

* 📝 **W&B Report**:
  *https://api.wandb.ai/links/shreyadhondi-indian-institute-of-technology-madras/13ytftex*

---

## 👤 Author

**Shreya Dhondi**
DA24M019
Indian Institute of Technology Madras

---
