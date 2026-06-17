import argparse
from pathlib import Path

import torch
import yaml
from tqdm import tqdm

from compute.computeAccuracy import test
from compute.forward_backward import update_parameter
from models.module import YOLOv8Loss
from optimizer.lr_policy import CosineLRwithWarmup
from optimizer.optimizer_scale import Adamwscale_2, get_maxGrad
from utils.dataset import getDataset
from utils.getModel import getModel, get_net_parameters


def write_runtime_config(dataset, modelname, epoch, g_bits, maxG_conv, maxG_fc):
    """Write optimizer settings used by the gradient scaling utilities."""
    config_dir = Path("config")
    config_dir.mkdir(parents=True, exist_ok=True)

    data = {
        "method": "GSC_AL",
        "dataset": dataset,
        "modelname": modelname,
        "gbit": g_bits,
        "epoch": epoch,
        "maxGrad_conv": maxG_conv,
        "maxGrad_fc": maxG_fc,
    }
    with (config_dir / "config.yaml").open("w", encoding="utf-8") as file:
        yaml.safe_dump(data, file, sort_keys=False)


def run(dataset, size, learningRateQ, batchsize, epoch, w_bits, a_bits, g_bits, modelname, maxG_conv, maxG_fc, device):
    train_loader, test_loader = getDataset(dataset, size=size, batchsize=batchsize, testbatchsize=batchsize)

    tlayer = 8
    strides = [8, 16, 32]
    num_class = 1

    computeLoss = YOLOv8Loss(num_classes=num_class, imagesize=size, reg_max=16, topk=10, strides=strides, device=device)

    model = getModel(num_class, modelname, signed=True, w_bits=w_bits, a_bits=a_bits)
    model.to(device=device)

    write_runtime_config(dataset=dataset, modelname=modelname, epoch=epoch, g_bits=g_bits, maxG_conv=maxG_conv, maxG_fc=maxG_fc)
    update_parameter()
    get_maxGrad()

    model.quantize()
    model.to(device=device)
    model.eval()

    with torch.no_grad():
        for i, (data, _) in enumerate(train_loader, 1):
            data = [torch.unsqueeze(t, dim=0) for t in data]
            data = torch.cat(data, dim=0).to(device)
            _ = model.quantize_forward(data)

            if i % 10 == 0:
                break

    model.freeze()
    print("Get quantized model")

    q_parameters = get_net_parameters(model, [])
    for name, parameter in model.named_parameters():
        if "head" in name:
            q_parameters.append(parameter)

    optimizerQ = Adamwscale_2(q_parameters, lr=learningRateQ, betas=(0.9, 0.999), weight_decay=5e-4)

    schedulerQ = CosineLRwithWarmup(optimizerQ, 5, learningRateQ * 0.1, epoch, final_lr=learningRateQ * 0.01)

    fitness = 0
    maxIndex = {}
    for e in range(1, epoch + 1):
        pbar = tqdm(train_loader, total=len(train_loader), leave=True)
        model.eval()

        for data, targets in pbar:
            data = [torch.unsqueeze(t, dim=0) for t in data]
            data = torch.cat(data, dim=0).to(device)

            record_act = []
            cls_out, reg_out = model.quantize_training(data, record_act)

            loss, loss_ = computeLoss(cls_out, reg_out, targets)
            loss_act = torch.tensor(0, dtype=torch.float, device=device)

            for act in record_act:
                mask = act.abs() > (2 ** (a_bits - 1)) * 0.9
                beta = mask.sum() / act.nelement()
                loss_act += act.abs().mean() * beta

            loss = loss + loss_act

            optimizerQ.zero_grad()
            loss.backward()

            optimizerQ.pre_step(model, e, epoch, tlayer)
            optimizerQ.step()
            model.update_quantize_parameters()

            pbar.set_description(f"E {e}")
            pbar.set_postfix(
                {
                    "loss": loss.item(),
                    "clsloss": loss_[0].item(),
                    "iouloss": loss_[1].item(),
                    "dflloss": loss_[2].item(),
                    "actloss": loss_act.item(),
                    "mem GPU": torch.cuda.max_memory_allocated() / 1024**3,
                }
            )

        schedulerQ.step()

        should_test = e % 5 == 0 or (e >= int(epoch * 0.9) and e % 1 == 0)
        if should_test:
            index = test(model, dataset, "GSC_AL", num_class, test_loader, strides, size, e, device, plot=True)
            current_fitness = index["F1"] * 0.5 + index["AP"] * 0.5
            if not maxIndex or fitness < current_fitness:
                fitness = current_fitness
                maxIndex = index

    print(maxIndex)
    return maxIndex


def parse_args():
    parser = argparse.ArgumentParser(description="Train quantized detection models with GSC-AL.")
    parser.add_argument("--dataset", type=str, default="UCAS", choices=["UCAS", "PDT"], help="Dataset name configured in utils/dataset.py.")
    parser.add_argument("--imgsize", type=int, default=448, help="Input image size.")
    parser.add_argument("--batchsize", type=int, default=4, help="Training batch size.")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs.")
    parser.add_argument("--lr", type=float, default=5e-2, help="Learning rate.")
    return parser.parse_args()


def main():
    opt = parse_args()

    w_bits = 8
    a_bits = 8
    g_bits = 8

    maxG_conv = 2 ** (w_bits - 1)
    maxG_fc = 2 ** (w_bits - 1)
    modelname = "resnet18"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    run(opt.dataset, opt.imgsize, opt.lr, opt.batchsize, opt.epochs, w_bits, a_bits, g_bits, modelname, maxG_conv=maxG_conv, maxG_fc=maxG_fc, device=device)


if __name__ == "__main__":
    main()
