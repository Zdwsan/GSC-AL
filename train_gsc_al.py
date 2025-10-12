
import torch.nn as nn
import torch
import yaml
import argparse

from optimizer.optimizer_scale import *
from tqdm import tqdm
from optimizer.lr_policy import CosineLRwithWarmup
from compute.computeAccuracy import *
from utils.getModel import getModel, get_net_parameters
from utils.dataset import *
from models.quantize_dequantize import *
from models.quantize_class import *
from models.module import *
from models.fpn import *
from compute.computeAccuracy import test
from compute.forward_backward import update_parameter

def run(dataset, size, learningRateQ, batchsize, epoch, w_bits, a_bits, g_bits, modelname, methods, maxG_conv, maxG_fc, device):

    train_loader, test_loader = getDataset(dataset, size=size, batchsize=batchsize, testbatchsize=batchsize)

    tlayer = 8

    strides = [8, 16, 32]

    num_class = 1

    computeLoss = YOLOv8Loss(num_classes=num_class, imagesize=size, reg_max=16, topk=10, strides=strides, device=device)

    model = getModel(num_class, modelname, signed=True, w_bits=w_bits, a_bits=a_bits)
    model.to(device=device)

    data = {
        'method' : methods,
        'dataset' : dataset,
        'modelname' : modelname,
        'gbit' : g_bits,
        'epoch': epoch,
        'maxGrad_conv': maxG_conv,
        'maxGrad_fc': maxG_fc,
    }
    yaml_str = yaml.dump(data)
    with open('config/config.yaml', 'w') as file:
        file.write(yaml_str)
    update_parameter()
    get_maxGrad()

    model.quantize( )
    model.to(device=device)
    model.eval()
    with torch.no_grad():
        for i, (data, target) in enumerate(train_loader, 1):
            data = [torch.unsqueeze(t,dim=0) for t in data]  # torch.unsqueeze增加一个维度
            data = torch.cat(data, dim=0)
            data = data.to(device)
            _ = model.quantize_forward(data)

            if i%10 == 0:
                break
    model.freeze()

    print('Get quantized model')

    QParameters = []
    QParameters = get_net_parameters(model, QParameters)

    for n, p in model.named_parameters():
        if 'head' in n:
            QParameters.append(p)


    if methods in ['QAS', 'QAS_AL']:
        optimizerQ = AdamwScale(
            QParameters,
            lr = learningRateQ,
            betas=(0.9, 0.999),
            weight_decay=5e-4
            )

    elif methods in ['GSC', 'GSC_AL']:
        optimizerQ = Adamwscale_2(
            QParameters,
            lr = learningRateQ,
            betas=(0.9, 0.999),
            weight_decay=5e-4
            )
        

    schedulerQ = CosineLRwithWarmup(
        optimizerQ, 5, learningRateQ*0.1,
        epoch,
        final_lr=learningRateQ*0.01
    )

    fitness = 0
    for e in range(1, epoch+1):

        pbar = tqdm(train_loader, total=len(train_loader), leave=True)

        model.eval()

        for i, (data, targets) in enumerate(pbar, 1):

            data = [torch.unsqueeze(t,dim=0) for t in data]  # torch.unsqueeze增加一个维度
            data = torch.cat(data, dim=0)
            data = data.to(device)

            record_act = []
            cls_out, reg_out = model.quantize_training(data, record_act)

            loss, loss_ = computeLoss(cls_out, reg_out, targets)
            loss_act = torch.tensor(0, dtype=torch.float).to(device)

            if methods in ['GSC_AL', 'QAS_AL']:
                for ct, act in enumerate(record_act):
                    mask = (act.abs() > (2**(a_bits-1)) * 0.9 ) 
                    mask = mask.sum()
                    beta = mask / act.nelement()
                    loss_act += act.abs().mean() * beta
   
                loss = loss + loss_act 

            elif methods in ['GSC', 'QAS']:
                for act in record_act:
                    act = act.detach()
                    mask = (act.abs() > (2**(a_bits-1)) * 0.9 ) 
                    mask = mask.sum()
                    beta = mask / act.nelement()
                    loss_act += act.abs().mean() * beta
                loss = loss

            optimizerQ.zero_grad()
            loss.backward()
            
            if methods in ['GSC_AL', 'GSC'] :
                optimizerQ.pre_step(model, e, epoch, tlayer )
            elif methods in ['QAS', 'QAS_AL']:
                optimizerQ.pre_step(model)

            optimizerQ.step()

            model.update_quantize_parameters()  # update scale

            pbar.set_description('E {}'.format(e))
            pbar.set_postfix({
                            "loss": loss.item(),
                            "clsloss": loss_[0].item(),
                            "iouloss": loss_[1].item(),
                            "dflloss": loss_[2].item(),
                            "actloss": loss_act.item(),
                            "mem GPU": torch.cuda.max_memory_allocated() / 1024**3
            })

        schedulerQ.step()

        if e%5==0 or (e>=int(epoch*0.9) and e%1==0):
            index = test(model, dataset, methods, num_class, test_loader, strides, size, e, device, plot=True)
            print(index)
  
    return model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", type=str, default="GSC_AL")
    parser.add_argument("--dataset", type=str, default="UCAS")
    parser.add_argument("--imgsize", type=int, default=448)
    parser.add_argument("--batchsize", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=5e-2)
    opt = parser.parse_args()

    dataset = opt.dataset
    size = opt.imgsize
    lrQ = opt.lr
    batchsize = opt.batchsize
    epoch = opt.epochs
    methods = opt.method

    w_bits = 8
    a_bits = 8
    g_bits = 8
    
    maxG_conv = 2**(w_bits-1) 
    maxG_fc = 2**(w_bits-1) 

    modelname = 'resnet18'

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = run(dataset, size, lrQ, batchsize, epoch, w_bits, a_bits, g_bits, modelname, methods, 
                              maxG_conv=maxG_conv, maxG_fc=maxG_fc, device=device)


if __name__=='__main__':
    main()