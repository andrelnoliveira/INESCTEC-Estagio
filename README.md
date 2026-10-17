# INESC TEC — UAV Obstacle Detection & Measurement

> **Status:** Work in progress — SegFormer and SSD integration complete. Hybrid pipeline combining both models is functional; final model weights and evaluation results pending.

This project focuses on the **detection and measurement of obstacles** using an Unmanned Aerial Vehicle (UAV). It combines semantic segmentation (SegFormer) with object detection (SSD MobileNet V1) to produce bounding boxes and spatial measurements from aerial imagery.

## Pipeline Overview

```
Input Image
    │
    ▼
┌─────────────┐
│  SegFormer  │  ──►  Semantic segmentation map
└─────────────┘       (classifies every pixel into categories)
    │
    ▼
┌─────────────┐
│     SSD     │  ──►  Bounding boxes around detected obstacles
└─────────────┘       (Single Shot MultiBox Detector)
    │
    ▼
Obstacle Measurements
(size, position, and distance estimates from UAV perspective)
```

---

## Requirements

| Dependency     | Version   | Purpose                                      |
|----------------|-----------|----------------------------------------------|
| Python         | 3.9       | Runtime environment                          |
| Anaconda       | Latest    | Environment and package management           |
| PyTorch        | 2.6.0       | Deep learning framework (SegFormer)          |
| CUDA           | 11.8      | GPU acceleration (required for training)     |
| Torchvision    | 0.22.1 | Data augmentation and weather simulation |
| TensorFlow     | 2.x       | Deep learning framework (SSD MobileNet V1)   |
| Git            | Latest    | Version control and repository cloning       |
| Weights & Biases (WandB) | Latest | Training metrics logging and experiment tracking |

> **Note on WandB:** WandB (Weights & Biases) is used to log and visualise training metrics such as loss, accuracy, and validation scores in real time. Without logging in, training will still run but no metrics will be recorded remotely. You can create a free account at [wandb.ai](https://wandb.ai).

---

## Installation

### 1. Clone the repository

```bash
git clone <repository-url>
cd inesctec-internership
```

### 2. Automated setup (recommended)

Choose the script for your operating system:

**Linux / macOS:**
```bash
bash install.sh
```

**Windows (PowerShell):**
```powershell
./install.ps1
```

### 3. Manual setup

If you prefer to install manually:

```bash
conda create -n uavid python=3.9
conda activate uavid
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

---

## Folder Structure

```
Semantic_Segmentation_UAVid/
│
├── config/                  # Configuration files for each model
├── docker/                  # Dockerfile for containerised deployment
├── model_weights/           # Pre-trained weights (UAVid dataset)
├── src/                     # Model and dataset source code
├── utils/                   # Utility scripts (e.g., dataset preprocessing)
│
├── uavid/
│   ├── uavid_train/         # Original training set
│   │   ├── seq1/
│   │   │   ├── Images/
│   │   │   └── Labels/
│   │   └── ...
│   ├── uavid_val/           # Original validation set
│   │   ├── seq16/
│   │   │   ├── Images/
│   │   │   └── Labels/
│   │   └── ...
│   ├── uavid_test/          # Original test set (no labels)
│   │
│   ├── train/               # Preprocessed training set (patched)
│   │   ├── images/
│   │   └── masks/
│   └── val/                 # Preprocessed validation set (patched)
│       ├── images/
│       └── masks/
│
├── inference.py
├── train.py
├── hybrid_detection.py       # Combines SegFormer + SSD outputs
├── evaluation.py             # Precision, recall, F1, IoU, mAP metrics
└── requirements.txt
```

---

## Data Processing

Because the original UAVid images have very high resolution, training directly on full images is computationally infeasible. To address this, the dataset is split into 1024×1024 patches. This also re-labels the masks to match the expected class format.

Run the following for **both** the training and validation sets:

**Training set:**
```bash
python utils/uavid_patch_split.py \
  --input-dir "./uavid/uavid_train" \
  --output-img-dir "./uavid/train/images" \
  --output-mask-dir "./uavid/train/masks" \
  --split-size-h 1024 --split-size-w 1024 \
  --stride-h 1024 --stride-w 1024
```

**Validation set:**
```bash
python utils/uavid_patch_split.py \
  --input-dir "./uavid/uavid_val" \
  --output-img-dir "./uavid/val/images" \
  --output-mask-dir "./uavid/val/masks" \
  --split-size-h 1024 --split-size-w 1024 \
  --stride-h 1024 --stride-w 1024
```

Make sure the `uavid/` dataset folder is placed inside the repository root before running these commands.

---

## Models

### SegFormer
A transformer-based model used for semantic segmentation. It classifies each pixel of the input image into one of the UAVid categories (e.g., building, road, vegetation, vehicle, human, background).

### SSD MobileNet V1 — Single Shot MultiBox Detector
After segmentation, the SSD MobileNet V1 model (implemented in TensorFlow) generates bounding boxes around detected obstacles. These bounding boxes, combined with UAV altitude and camera parameters, enable spatial measurement of each obstacle.

### Hybrid Detection Pipeline
The `hybrid_detection.py` script integrates the SegFormer segmentation output with the SSD bounding box predictions, using K-means clustering to reconcile the two sources of information and produce a unified obstacle map. This combines the pixel-level accuracy of SegFormer with the fast, discrete object localisation of SSD.

---

## Training

Before training, you must authenticate with WandB. This allows the training run to upload logs, loss curves, and evaluation metrics to your WandB dashboard — making it easy to monitor progress remotely and compare experiments.

```bash
conda activate uavid
wandb login
```

You will be prompted to enter your WandB API key, which can be found at [wandb.ai/authorize](https://wandb.ai/authorize).

To start training, pass the path to a model configuration file:

```bash
python train.py -c config/Segformer.py
```

| Argument | Description                                      |
|----------|--------------------------------------------------|
| `-c`     | Path to the model configuration file to use     |

If you do not log in to WandB, training will still proceed, but metrics will only be available locally.

---

## Inference

The inference pipeline supports **ensemble learning**: you can pass one or more models along with a weight for each, controlling how much each model contributes to the final prediction.

```bash
python inference.py \
  -i ./uavid/uavid_val \
  -c ./config/Segformer.py \
  -w 1 \
  -o ./results/ \
  -ph 1024 -pw 1024 \
  -b 2 \
  -m
```

| Argument | Description                                                                 |
|----------|-----------------------------------------------------------------------------|
| `-i`     | Path to the input directory                                                 |
| `-c`     | Path to one or more model configuration files                               |
| `-w`     | Importance weight for each model (must match the number of `-c` arguments)  |
| `-o`     | Path to the output directory                                                |
| `-ph`    | Patch height                                                                |
| `-pw`    | Patch width                                                                 |
| `-b`     | Batch size                                                                  |
| `-m`     | Flag indicating the input set contains ground-truth masks for evaluation    |

> **Note:** The test set (`uavid_test`) does not contain labels, so omit the `-m` flag when running inference on it.

### SSD Training

The SSD MobileNet V1 model is trained separately using TensorFlow. Make sure your TensorFlow environment is active before running the training script:

```bash
python ssd_train.py -c config/ssd_mobilenet_v1.config
```

> **Note:** Refer to the SSD configuration file for details on checkpoint paths, number of classes, and training hyperparameters.

### Hybrid Detection

Once both the SegFormer and SSD models have produced their outputs, run the hybrid detection script to combine them into a single obstacle map:

```bash
python hybrid_detection.py \
  -i ./uavid/uavid_val \
  -seg-weights ./model_weights/segformer.pth \
  -ssd-weights ./model_weights/ssd_mobilenet_v1 \
  -o ./results/hybrid/
```

This script uses K-means clustering to merge the segmentation masks and the SSD bounding boxes into a consolidated set of detected obstacles.

---

## Evaluation

The `evaluation.py` script computes standard detection and segmentation metrics — **precision**, **recall**, **F1-score**, **IoU**, and **mAP** — by comparing model predictions against ground-truth labels:

```bash
python evaluation.py \
  -p ./results/hybrid/ \
  -gt ./uavid/uavid_val \
  -o ./results/metrics/
```

| Argument | Description                                      |
|----------|--------------------------------------------------|
| `-p`     | Path to the predictions to evaluate               |
| `-gt`    | Path to the ground-truth labels                   |
| `-o`     | Path to the output directory for the metrics report |
