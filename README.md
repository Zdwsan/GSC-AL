# GSC-AL

Official implementation of **Gradient Scale Correction and Activation Loss (GSC-AL)** for training quantized object detection models on private edge-side data.

## Abstract

Deploying pre-trained models on edge devices often requires quantization because of limited computation and memory resources. Adapting these models to private data usually relies on uploading data for retraining or fine-tuning, which can introduce security and privacy risks. This project implements Gradient Scale Correction (GSC), which rescales gradients to improve weight updates in quantized models, and Activation Loss (AL), which mitigates non-Gaussian activation distributions caused by accumulated quantization errors and activation truncation during training.

## Environment

The code has been tested on Ubuntu 20.04 with Python 3.11.

```bash
conda create --name gsc-al python=3.11
conda activate gsc-al
pip install -r requirements.txt
```

## Repository Structure

```text
.
├── compute/                 # Accuracy, matching, and forward/backward utilities
├── config/                  # Runtime configuration written by training scripts
├── models/                  # FP32 and quantized model definitions
├── optimizer/               # Learning-rate policy and quantization-aware optimizers
├── utils/                   # Dataset loading, categories, loss helpers
├── train_fp.py              # Train the FP32 baseline
├── train_qat.py             # Train the standard QAT baseline
└── train_gsc_al.py          # Train quantized models with GSC-AL
```

## Dataset Layout

The current dataset loader expects YOLO-format labels under `../dataset` relative to this repository. Supported dataset names are `UCAS` and `PDT`.

Example layout for UCAS:

```text
../dataset/UCAS_AOD/plane/
├── train/
│   ├── images/
│   └── labels/
└── val/
    ├── images/
    └── labels/
```

Each label file should be a `.txt` file with one object per line:

```text
class_id x_center y_center width height
```

Coordinates are expected to be normalized to `[0, 1]`.

## Training

```bash
python train_gsc_al.py --dataset UCAS --imgsize 448 --batchsize 32 --epochs 100 --lr 5e-2
```


## Citation

If this code is useful for your research, please cite the corresponding paper. Replace the placeholder below with the final BibTeX entry after publication:

```bibtex
@article{gscal,
  title  = {Gradient Scale Correction and Activation Loss for Quantized Model Training},
  author = {Your Name and Coauthors},
  year   = {2025}
}
```
