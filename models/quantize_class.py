import torch
import torch.nn as nn
import numpy as np
from models.quantize_dequantize import *
import torch.nn.functional as F
from torch.autograd import Variable
import copy
import yaml
from compute.forward_backward import _QAddFunct, _QAvgpoolFunc, _QConv2dFunc, _QLinearFunc, _QMaxpoolFunc, _QTruncate

global is_quant_grad
def get_isQuanGrad_config(env):
    global is_quant_grad
    is_quant_grad = env


class QLinear(QModule):
    def __init__(self, fc_module, w_bits, a_bits, qi=True, qo=True, signed=True, location=None, location_w=None, activate=False ):
        super(QLinear, self).__init__(qi=qi, qo=qo, a_bits=a_bits, signed=signed)
        self.w_bits = w_bits
        self.a_bits = a_bits
        self.signed = signed
        self.fc_module = fc_module
        self.location = location
        self.location_w = location_w
        self.qw = QParam(a_bits=w_bits, signed=signed, isWeight=True)
        self.register_buffer('M', torch.tensor(0))
        self.activate = activate

        self.QTrunc = _QTruncate.apply
        self.QLinear = _QLinearFunc.apply
 
    def freeze(self, qi=None, qo=None):
        if qi is not None:
            self.qi = qi
        if qo is not None:
            self.qo = qo

        self.M.data = (self.qw.scale * self.qi.scale / self.qo.scale).data.view(1, -1)

        self.fc_module.weight.data = self.qw.quantize_tensor(self.fc_module.weight.data)
        self.qw.zero_point = self.qw.zero_point.view(-1, 1)

        self.fc_module.bias.data = torch.nn.Parameter(quantize_tensor(self.fc_module.bias.data,
                                            scale=self.qw.scale * self.qi.scale,
                                            zero_point=0,
                                            num_bits=32, signed=True))
        
    def update_quantize_parameters(self):
        self.M, self.fc_module = self.qw.update_quantize_parameters(self.fc_module, self.M, self.qi, self.qo)

    def clamp_parameters(self):
        self.fc_module.weight.data.round()
        self.fc_module.bias.data.round()

    def forward(self, x):
        if hasattr(self, 'qi'):
            self.qi.update(x)
        
        self.qw.update(self.fc_module.weight.data)

        x = F.linear(x, 
                     self.fc_module.weight,
                     self.fc_module.bias)
        
        if self.activate:
            x = F.relu(x)

        if hasattr(self, 'qo'):
            self.qo.update(x)
        return x


    def quantize_inference(self, x):
        i = 2
        e = 3
        out = self.QLinear(x, self.fc_module.weight, self.fc_module.bias, 
                           self.qi, self.qo, self.qw.scale, self.qw.zero_point, 
                           self.M, self.signed, self.w_bits, i, e, self.location)
        out = self.QTrunc(out, self.a_bits, self.signed, self.location)
        return out
    

    def quantize_training(self, x, i, e):
        out = self.QLinear(x, self.fc_module.weight, self.fc_module.bias, self.qi, self.qo, self.qw.scale, self.qw.zero_point, self.M, self.signed, self.w_bits, i, e, self.location)
        out = self.QTrunc(out, self.a_bits, self.signed, self.location)
        return out

class QConvBNReLU(QModule):
    def __init__(self, conv_module, bn_module=None, w_bits=8, a_bits=8, qi=True, qo=True, activate=True, signed=True, bl=None):
        super(QConvBNReLU, self).__init__(qi=qi, qo=qo, a_bits=a_bits, signed=signed)
        self.w_bits = w_bits
        self.a_bits = a_bits
        self.conv_module = conv_module
        self.bn_module = bn_module
        self.signed = signed
        self.qw = QParam(a_bits=w_bits, signed=signed, isWeight=True)
        self.register_buffer('M', torch.tensor([], requires_grad=False))  
        self.activate = activate
        self.bl = bl

        self.QConv2d = _QConv2dFunc.apply
        self.QTrunc = _QTruncate.apply

    def fold_bn(self, conv, mean, std):
        if self.bn_module.affine:
            gamma_ = self.bn_module.weight / std
            weight = conv.weight * gamma_.view(conv.out_channels, 1, 1, 1)
            if conv.bias is not None:
                bias = gamma_ * conv.bias - gamma_ * mean + self.bn_module.bias
            else:
                bias = self.bn_module.bias - gamma_ * mean
        else:
            gamma_ = 1 / std
            weight = conv.weight * gamma_
            if conv.bias is not None:
                bias = gamma_ * conv.bias - gamma_ * mean
            else:
                bias = -gamma_ * mean
            
        return weight, bias

    def forward(self, x):
        if hasattr(self, 'qi'):
            self.qi.update(x )

        if self.bn_module is not None:
            if self.training:
                y = F.conv2d(x, self.conv_module.weight, self.conv_module.bias, 
                                stride=self.conv_module.stride,
                                padding=self.conv_module.padding,
                                dilation=self.conv_module.dilation,
                                groups=self.conv_module.groups)
                y = y.permute(1, 0, 2, 3) # NCHW -> CNHW
                y = y.contiguous().view(self.conv_module.out_channels, -1) # CNHW -> C,NHW
                mean = y.mean(1)
                var = y.var(1)
                self.bn_module.running_mean = \
                    (1 - self.bn_module.momentum) * self.bn_module.running_mean + \
                    self.bn_module.momentum * mean.detach()
                self.bn_module.running_var = \
                    (1 - self.bn_module.momentum) * self.bn_module.running_var + \
                    self.bn_module.momentum * var.detach()
            else:
                mean = Variable(self.bn_module.running_mean)
                var = Variable(self.bn_module.running_var)

            std = torch.sqrt(var + self.bn_module.eps)

            weight, bias = self.fold_bn(self.conv_module, mean, std)
        else:
            weight, bias = self.conv_module.weight, self.conv_module.bias  

        self.qw.update(weight.data)

        x = F.conv2d(x, 
                     FakeQuantize.apply(weight, self.qw), 
                     bias, 
                     stride=self.conv_module.stride,
                     padding=self.conv_module.padding, dilation=self.conv_module.dilation, 
                     groups=self.conv_module.groups) 
        
        if self.activate:
            x = F.relu(x)

        if hasattr(self, 'qo'):
            self.qo.update(x )
 
        return x

    def freeze(self, qi=None, qo=None):
        if qi is not None:
            self.qi = qi
        if qo is not None:
            self.qo = qo

        self.M.data = (self.qw.scale * self.qi.scale / self.qo.scale).data.view(1, -1, 1, 1)

        if self.bn_module is not None:
            std = torch.sqrt(self.bn_module.running_var + self.bn_module.eps)

            weight, bias = self.fold_bn(self.conv_module, self.bn_module.running_mean, std)
        else:
            weight, bias = self.conv_module.weight, self.conv_module.bias
        self.conv_module.weight.data = self.qw.quantize_tensor(weight.data)
        self.qw.zero_point = self.qw.zero_point.view(-1, 1, 1, 1)
        
        self.conv_module.bias = torch.nn.Parameter(quantize_tensor(bias, scale=self.qi.scale * self.qw.scale,
                                                    zero_point=0, num_bits=32, signed=True))
            
    def update_quantize_parameters(self):
        self.M, self.conv_module = self.qw.update_quantize_parameters(self.conv_module, self.M, self.qi, self.qo)

    def clamp_parameters(self):
        self.conv_module.weight.data.round_()
        self.conv_module.bias.data.round_()

    def quantize_inference(self, x):
        out = self.QConv2d(x, self.conv_module.weight, self.conv_module.bias, self.conv_module.stride, 
                                 self.conv_module.padding, self.conv_module.dilation, self.conv_module.groups, 
                                 self.qi, self.qo, self.qw.scale, self.qw.zero_point, self.M, self.signed, self.w_bits)
        out = self.QTrunc(out, self.a_bits, self.signed)
        return out
    

    def quantize_training(self, x):
        out = self.QConv2d(x, self.conv_module.weight, self.conv_module.bias, self.conv_module.stride, 
                                 self.conv_module.padding, self.conv_module.dilation, self.conv_module.groups, 
                                 self.qi, self.qo, self.qw.scale, self.qw.zero_point, self.M, self.signed, self.w_bits)

        out = self.QTrunc(out, self.a_bits, self.signed)
        return out

        

class QAddReLU(QModule):
    def __init__(self, a_bits, qi1=None, qi2=None, qo=None, signed=True, activate=False):
        super(QAddReLU, self).__init__(qo=qo, a_bits=a_bits, signed=signed)
        self.a_bits = a_bits
        self.signed = signed
        self.activate = activate
        self.register_buffer('M1', torch.tensor([]))
        self.register_buffer('M2', torch.tensor([]))
        self.QTrunc = _QTruncate.apply
        self.QAdd = _QAddFunct.apply

    def freeze(self, qi1=None, qi2=None, qo=None):
        if qi1 is not None:
            self.qi1 = qi1
        if qi2 is not None:
            self.qi2 = qi2
        if qo is not None:
            self.qo = qo       
        
        self.M1.data = (self.qi1.scale / self.qo.scale).data
        self.M2.data = (self.qi2.scale / self.qo.scale).data

    def forward(self, x1, x2):
        if hasattr(self, 'qi1'):
            self.qi1.update(x1 )
        if hasattr(self, 'qi2'):
            self.qi2.update(x2 )
        x = x1 + x2
        
        if self.activate:
            x = F.relu(x)
        
        if hasattr(self, 'qo'):
            self.qo.update(x )
        return x
    
    def quantize_inference(self, x1, x2):
        x1 = x1
        x2 = x2
        out = self.QAdd(x1, x2, self.qi1, self.qi2, self.qo)
        out = self.QTrunc(out, self.a_bits, self.signed)
        return out


    def quantize_training(self, x1, x2):
        x1 = x1
        x2 = x2
        out = self.QAdd(x1, x2, self.qi1, self.qi2, self.qo)
        out = self.QTrunc(out, self.a_bits, self.signed)
        return out

    
class QAvgPooling2d(QModule):
    def __init__(self, a_bits, kernel_size=7, stride=1, padding=0, qi=False, qo=True, signed=True):
        super(QAvgPooling2d, self).__init__(qi=qi, a_bits=a_bits, signed=signed)
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.a_bits = a_bits
        self.signed = signed
        self.register_buffer('M', torch.tensor(0))
        self.QAvgp = _QAvgpoolFunc.apply
        self.QTrunc = _QTruncate.apply
 

    def freeze(self, qi=None, qo=None):
        if qi is not None:
            self.qi = qi
        if qo is not None:
            self.qo = qo

        self.M.data = (self.qi.scale / self.qo.scale).data
        
    def forward(self, x):
        if hasattr(self, 'qi'):
            self.qi.update(x)
        x = F.avg_pool2d(x, self.kernel_size, self.stride, self.padding)
    
        if hasattr(self, 'qo'):
            self.qo.update(x)
        return x    
    
    def quantize_inference(self, x):
        out = self.QAvgp(x, self.kernel_size, self.stride, self.padding, self.qi, self.qo)
        out = self.QTrunc(out, self.a_bits, self.signed)
        return out
    
    def quantize_training(self, x):
        out = self.QAvgp(x, self.kernel_size, self.stride, self.padding, self.qi, self.qo)
        out = self.QTrunc(out, self.a_bits, self.signed)
        return out


class QMaxPooling2d(QModule):
    def __init__(self, a_bits, kernel_size=3, stride=1, padding=0, qi=False, qo=True, signed=True):
        super(QMaxPooling2d, self).__init__(qi=qi, a_bits=a_bits, signed=signed)
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.a_bits = a_bits
        self.signed = signed
        self.register_buffer('M', torch.tensor(0))
        self.QTrunc = _QTruncate.apply
 
    def freeze(self, qi=None, qo=None):

        if qi is not None:
            self.qi = qi
        if qo is not None:
            self.qo = qo

        self.M.data = (self.qi.scale / self.qo.scale).data
 
    def forward(self, x):
        if hasattr(self, 'qi'):
            self.qi.update(x)

        x = F.max_pool2d(x, self.kernel_size, self.stride, self.padding)
    
        if hasattr(self, 'qo'):
            self.qo.update(x)
        return x    

    def quantize_inference(self, x):
        x = STE.apply(x)
        x = x - self.qi.zero_point
        x = F.max_pool2d(x, self.kernel_size, self.stride, self.padding)
        x = self.M * x
        x = STE.apply(x)
        out = x + self.qo.zero_point
        out = self.QTrunc(out, self.a_bits, self.signed)
        return out
    
    def quantize_training(self, x):
        x = STE.apply(x)
        x = x - self.qi.zero_point
        x = F.max_pool2d(x, self.kernel_size, self.stride, self.padding)
        x = self.M * x
        x = STE.apply(x)
        out = x + self.qo.zero_point
        out = self.QTrunc(out, self.a_bits, self.signed)
        return out
