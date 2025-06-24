
import torch.nn as nn
import math
from models.resnet18 import *
from models.quantize_class import *


class myResidual(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels, stride):
        super(myResidual, self).__init__()
        self.ineqout = (in_channels == out_channels) and (stride == 1)
        self.in_channels = in_channels
        self.mid_channels = mid_channels
        self.out_channels = out_channels
        self.stride = stride
        self.layer1 = block(in_channels, mid_channels, kernel_size=1, stride=1, padding=0, groups=1)
        self.layer2 = block(mid_channels, mid_channels, kernel_size=3, stride=stride, padding=1, groups=mid_channels)
        self.layer3 = block(mid_channels, out_channels, kernel_size=1, stride=1, padding=0, groups=1, activate=False)
        if self.ineqout is not True:
            self.shortcut = block(in_channels, out_channels, kernel_size=1, stride=stride, padding=0, groups=1, activate=False)
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)      
        if self.ineqout:
            out = x + out
        else:
            out = self.shortcut(x) + out
        return out
    



class YOLOv8Head(nn.Module):
    def __init__(self, in_channels, num_classes=80, reg_max=16, img_size=448, strides=None):
        super().__init__()
        self.num_classes = num_classes
        self.reg_max = reg_max
        self.img_size = img_size
        self.strides = strides

        self.cls_out = nn.Conv2d(in_channels, num_classes, 1)
        self.reg_out = nn.Conv2d(in_channels, 4 * reg_max, kernel_size=1)

    def forward(self, x):
        cls_out = self.cls_out(x)
        reg_out = self.reg_out(x)
        return cls_out, reg_out
    
    def bias_init(self):
        m = self  
        self.cls_out.bias.data[:] = 1.0  # box
        self.reg_out.bias.data[: self.num_classes] = math.log(5 / m.nc / (self.img_size / self.strides) ** 2)  # cls (.01 objects, 80 classes, 640 img)

class Resnet18_FPN(nn.Module):
    def __init__(self, num_classes=80, img_size=448, strides=[8, 16, 32]):
        super(Resnet18_FPN, self).__init__()

        self.net = resnet18()
        self.net.load_pre_parameters()

        self.neck5 = myResidual(512, 1024, 256, stride=1)
        self.neck4 = myResidual(256, 512, 256, stride=1)
        self.neck3 = myResidual(128, 256, 256, stride=1)

        self.head5 = YOLOv8Head(256, num_classes=num_classes, reg_max=16, img_size=img_size, strides=strides[2])
        self.head4 = YOLOv8Head(256, num_classes=num_classes, reg_max=16, img_size=img_size, strides=strides[1])
        self.head3 = YOLOv8Head(256, num_classes=num_classes, reg_max=16, img_size=img_size, strides=strides[0])

        self.up4= nn.Upsample(scale_factor=2)
        self.up3= nn.Upsample(scale_factor=2)

    def forward(self, x):
        features = self.net(x) 
   
        p5 = self.neck5(features[3])  
        p4 = self.neck4(features[2]) + self.up4(p5)
        p3 = self.neck3(features[1]) + self.up3(p4)

        cls_out, reg_out = [], []
        heads = [self.head3, self.head4, self.head5]
        for head, feature in zip(heads, [p3, p4, p5]):
            cls, reg = head(feature)
            cls_out.append(cls), reg_out.append(reg)
        return cls_out, reg_out
    
