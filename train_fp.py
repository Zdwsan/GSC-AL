
import torch.nn.functional as F
import torch.nn as nn
import torch
import argparse

from torch.utils.data import Dataset
from torchvision import transforms
from tqdm import tqdm
from optimizer.lr_policy import CosineLRwithWarmup
from models.fpn import *
from utils.dataset import *
from models.module import *
from torch.amp import GradScaler, autocast
from compute.computeAccuracy import test


def get_wNorm_gNorm(model, wList, gList):
    for n, m in model.named_children():
            if isinstance(m, nn.Conv2d):
                if m.weight.grad is not None:
                    wList.append(m.weight.norm().detach().cpu().numpy()) 
                    gList.append(m.weight.grad.norm().detach().cpu().numpy()) 

            elif isinstance(m, nn.Linear):
                if m.weight.grad is not None:
                    wList.append(m.weight.norm().detach().cpu().numpy()) 
                    gList.append(m.weight.grad.norm().detach().cpu().numpy()) 

            elif isinstance(m, (nn.Module, nn.ModuleList, nn.Sequential)):
                get_wNorm_gNorm(m, wList, gList)
    return wList, gList

class dataSet(Dataset):
    def __init__(self, data, label):
        self.datas =  data.clone()
        self.labels = label.clone()

    def __len__(self):
        return len(self.datas)
        
    def __getitem__(self, idx):
        data = self.datas[idx].clone()
        label = self.labels[idx].clone()
        return data, label

def computeACC(pred, targ):
    pred = F.softmax(pred, dim=1)
    preds = pred.argmax(dim=1) 
    acc = torch.mean((preds == targ).float())
    return acc


def test_fp32(model, test_loader, device):
    model.eval()  
    with torch.no_grad():
        total = 0
        acc = 0
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            preds = outputs.argmax(dim=1) 
            acc = acc + torch.sum(preds == labels)
            total = total + images.shape[0]
        print('Test Accuracy of FP32 model: {} %'.format(100 * acc / total))
    return 100 * acc / total

def test_quantize(model, test_loader, device):
    with torch.no_grad():
        model.set_eval()  
        model.freeze()
        total = 0
        acc = 0
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model.quantize_inference(images)
            preds = outputs.argmax(dim=1) 
            acc = acc + torch.sum(preds == labels)
            total = total + images.shape[0]
        print('imagenet inference acc: {} %'.format(100 * acc / total))
    # torch.save(model.state_dict(), 'mobilenet_quantization.pt')
    torch.save(model, 'mobilenet_quantization_model.pt')
    # for name, buf in model.named_parameters():
    #     print(name)

my_trans = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

def delete_bn(model):
    for n, m in model.named_children():
        if isinstance(m, nn.BatchNorm2d):
            m = nn.Identity()

        elif isinstance(m, (nn.Module, nn.ModuleList, nn.Sequential)):
            delete_bn(m)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="UCAS")
    parser.add_argument("--imgsize", type=int, default=448)
    parser.add_argument("--batchsize", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    opt = parser.parse_args()
    run(opt)
    

def run(opt):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    BATCH_SIZE = opt.batchsize
    epoch = opt.epochs
    size = opt.imgsize
    dataset = opt.dataset
    learningRate = opt.lr


    strides = [8, 16, 32]

    num_class = 1

    model = Resnet18_FPN(num_classes=num_class, img_size=size, strides=strides)

    train_loader, test_loader = getDataset(dataset, size=size, batchsize=BATCH_SIZE, testbatchsize=BATCH_SIZE)
    
    optimizer = torch.optim.AdamW(model.parameters(), learningRate, weight_decay=5e-4)

    computeLoss = YOLOv8Loss(num_classes=num_class, imagesize=size, reg_max=16, topk=10, 
                        strides=strides, device=device)

    scheduler = CosineLRwithWarmup(
            optimizer, 5, learningRate*0.1,
            epoch,
            final_lr=learningRate*0.1
        )
    
    model.to(device=device)

    scaler = GradScaler()
    max_map = {'AP':0}
    fitness = 0
    for e in range(1, epoch+1):
        model.train()
        pbar = tqdm(train_loader, total=len(train_loader), leave=True)
        for i, (data, targets) in enumerate(pbar, 1):
            data = [torch.unsqueeze(t,dim=0) for t in data]  # torch.unsqueeze增加一个维度
            data = torch.cat(data, dim=0)
            data = data.to(device)

            model.to(device)
            
            with autocast('cuda'):

                cls_out, reg_out = model(data)
                loss, loss_ = computeLoss(cls_out, reg_out, targets)

                optimizer.zero_grad()
                scaler.scale(loss).backward()


                scaler.step(optimizer)
            scaler.update()

            pbar.set_description('Train Epoch {}'.format(e))
            pbar.set_postfix({"loss": loss.item(),
                                "clsloss": loss_[0].item(),
                                "iouloss": loss_[1].item(),
                                "dflloss": loss_[2].item(),
                                })
        scheduler.step()
        if e%5==0 or (e>=int(epoch*0.9) and e%1==0):
            index = test(model, dataset, 'FP32', num_class, test_loader, strides, size, e, device, plot=True)
            print(index)


if __name__=='__main__':
    main()