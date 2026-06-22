# GSC-AL

## Abstract

The deployment of pre-trained models on edge devices often necessitates quantization due to computational resource constraints. Adapting these models to new private data typically involves data uploads for retraining or fine-tuning, which raises significant security concerns. Existing training methods struggle to directly train quantized models on edge devices. To address this, we analyzed the training process of quantized models and proposed a novel training method based on Gradient Scale Correction and Activation Loss (GSC-AL). Initially, we focused on the gradient and weight mismatch in quantized models during training. The proposed Gradient Scale Correction (GSC) efficiently scales the gradient to a defined range, enabling effective weight updates in quantized models. Furthermore, we introduced the activation Loss (AL) in the loss function, effectively alleviating the non-Gaussian activation distribution phenomenon caused by the accumulation of quantization errors and activation truncation problems in the training process based on GSC. Finally, we validate our method on seven common datasets using ResNet18 and MobileNetV2 as baseline models. Experimental results demonstrate that our GSC-AL method significantly enhances model predictive performance, achieving a training accuracy improvement of 41% compared to existing methods. The proposed GSC-AL method is presented at https://github.com/Zdwsan/GSC-AL.
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
@article{zhang2026directly,
  title={Directly Training on Quantized Model via Gradient Scale Correction for Edge Device},
  author={Zhang, Dewang and Yuan, Jingling and Zhou, Yu and Hu, Chuang and Yu, Xiaohan and Gou, Heping},
  journal={Neural Networks},
  pages={109266},
  year={2026},
  publisher={Elsevier}
}
```
