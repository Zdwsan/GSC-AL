import torch.nn as nn
import torchvision.models as Models
import torch
import torch.nn.functional as F
from pytorchcv.model_provider import get_model as ptcv_get_model

class block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, groups, activate=True):
        super(block, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, groups=groups, bias=False)  # bias False
        self.bn = nn.BatchNorm2d(out_channels, momentum=0.1)
        self.activate = activate
        if activate:
            self.relu = nn.ReLU()
    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        if self.activate:
            out = self.relu(out)
        return out
    
    def add_bias(self):
        temp = self.conv.weight.data
        self.conv = nn.Conv2d(self.conv.in_channels, self.conv.out_channels, 
                              kernel_size=self.conv.kernel_size, stride=self.conv.stride, 
                              padding=self.conv.padding, groups=self.conv.groups, bias=True)
        self.conv.weight.data = temp

class myInvertedResidual(nn.Module):
    def __init__(self, in_channels, out_channels, downsample=False):
        super(myInvertedResidual, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.downsample = downsample
        if downsample:
            self.layer1 = block(in_channels, out_channels, kernel_size=3, stride=2, padding=1, groups=1, activate=True)
        else:
            self.layer1 = block(in_channels, out_channels, kernel_size=3, stride=1, padding=1, groups=1, activate=True)
        self.layer2 = block(out_channels, out_channels, kernel_size=3, stride=1, padding=1, groups=1, activate=False)
        if downsample:
            self.shortcut = block(in_channels, out_channels, kernel_size=1, stride=2, padding=0, groups=1, activate=False)
        self.relu = nn.ReLU()
        
    def set_train(self):
        self.train()
        self.layer1.train()
        self.layer2.train()
    
    def set_eval(self):
        self.eval()
        self.layer1.eval()
        self.layer2.eval()

    def add_bias(self):
        self.layer1.add_bias()
        self.layer2.add_bias()

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        if self.downsample:
            out2 = self.shortcut(x)
            return self.relu(out + out2)
        else:
            return self.relu(out + x)
      
class resnet18(nn.Module):
    def __init__(self):
        super(resnet18, self).__init__()
        self.layer0 = block(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3, groups=1 )
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        inchannel = [64, 64,  128, 256]
        outchannel= [64, 128, 256, 512]
        nlayers = [2, 2, 2, 2]

        self.feature = nn.ModuleList([])
        self.flag = []
        for n, (i, o, nl) in enumerate(zip(inchannel, outchannel, nlayers)):
            for c in range(nl):
                self.flag.append(0)
                if n==0:
                    self.feature.append(myInvertedResidual(i, o, False))
                else:
                    if c==0:
                        self.feature.append(myInvertedResidual(i, o, True))
                    else:
                        self.feature.append(myInvertedResidual(o, o, False))
            self.flag[-1] = 1


    def forward(self, x):
        x = self.layer0(x)
        x = self.maxpool(x)
        out = []
        for layer, f in zip(self.feature, self.flag):
            x = layer(x)
            if f:
                out.append(x)
        return out
    

    def load_pre_parameters(self):
        pre_mobilenet = ptcv_get_model('resnet18', pretrained=True)
        new_state_dict = pre_mobilenet.state_dict()
        op = self.state_dict()
        for new_state_dict_num, new_state_dict_value in enumerate(new_state_dict.values()):
            for op_num, op_key in enumerate(op.keys()):
                if op_num == new_state_dict_num and op_num <= 100000:
                    if op[op_key].shape == new_state_dict_value.shape:
                        op[op_key] = new_state_dict_value                        
        self.load_state_dict(op)