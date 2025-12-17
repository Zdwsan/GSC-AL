# GSC-AL

## Abstract
The deployment of pre-trained models on edge devices often necessitates quantization due to computational resource constraints. Adapting these models to new private data typically involves data uploads for retraining or fine-tuning, which raises significant security concerns. Existing training methods struggle to directly train quantized models on edge devices. To address this, we analyzed the training process of quantized models and proposed a novel training method based on Gradient Scale Correction and Activation Loss (GSC-AL). Initially, we focused on the gradient and weight mismatch in quantized models during training. The proposed Gradient Scale Correction (GSC) efficiently scales the gradient to a defined range, enabling effective weight updates in quantized models. \textcolor{violet}{Furthermore, we introduced the activation Loss (AL) in the loss function, effectively alleviating the non-Gaussian activation distribution phenomenon caused by the accumulation of quantization errors and activation truncation problems in the training process based on GSC.} Finally, we validate our method on seven common datasets using ResNet18 and MobileNetV2 as baseline models. Experimental results demonstrate that our GSC-AL method significantly enhances model predictive performance, achieving a training accuracy improvement of 41\% compared to existing methods. \textcolor{red}{The proposed GSC-AL method is presented at https://github.com/Zdwsan/GSC-AL.

## OS
ubuntu 20.04

## Intall
conda create --name gsc python=3.11

conda activate gsc

pip install -r requirements.txt

## Training for FP32 model
python train_fp.py --dataset UCAS --imgsize 448 --batchsize 32 --epochs 100 --lr 1e-3

## Training for quantized model
python train_gsc_al.py --method GSC_AL --dataset UCAS --imgsize 448 --batchsize 32 --epochs 100 --lr 5e-2
