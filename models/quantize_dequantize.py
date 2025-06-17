import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Function

def get_quantize_limit(bits, signed):
    if signed:
        upper = 2 ** (bits - 1) - 1
        lower = - 2 ** (bits - 1) + 1
    elif signed==False:
        upper = 2 ** bits - 1
        lower = 0.
    return lower, upper

def calcScaleZeroPoint_W(min_val, max_val, num_bits, signed=True):
    # input: min_val, max_val: r_min, r_max
    absmax = torch.zeros_like(max_val)
    mask = max_val.abs() > min_val.abs()
    absmax[mask] = max_val.abs()[mask]
    mask = max_val.abs() < min_val.abs()
    absmax[mask] = min_val.abs()[mask]

    scale = absmax / (2. ** (num_bits-1) - 1)
    zero_point = torch.zeros_like(max_val)
    return scale, zero_point

def calcScaleZeroPoint(min_val, max_val, num_bits, signed=True):
    # input: min_val, max_val: r_min, r_max
    qmin, qmax = get_quantize_limit(num_bits, signed)
    scale = (max_val - min_val) / (qmax - qmin)
    
    if scale == 0:
        scale = torch.tensor(1e-4).to(min_val.device)
    #     print('error in calcScaleZeroPoint, scale = 0')
        zero_point = torch.tensor(0.).to(min_val.device)
    else:
        zero_point = qmax - max_val / scale

    if zero_point < qmin:
        zero_point = torch.tensor([qmin], dtype=torch.float32).to(min_val.device)
    elif zero_point > qmax:
        zero_point = torch.tensor([qmax], dtype=torch.float32).to(max_val.device)

    zero_point.round_()
   
    return scale, zero_point

def quantize_tensor_W(x, scale, zero_point, num_bits, signed=True):
    if signed:
        qmin = - 2. ** (num_bits - 1)
        qmax = 2. ** (num_bits - 1) - 1
    else:
        qmin = 0.
        qmax = 2. ** num_bits - 1.

    if scale.isnan().sum() != 0:
        print('test')

    mask0 = scale == 0
    scale[mask0] = 1e-10

    if len(x.shape) == 4:
        q_x = zero_point.view(-1, 1, 1, 1) + x / scale.view(-1, 1, 1, 1)
    elif len(x.shape) == 2:
        q_x = zero_point.view(-1, 1) + x / scale.view(-1, 1)
    q_x.clamp_(qmin, qmax).round_()
    # q_x = _roundSET.apply(q_x.clamp_(qmin, qmax))
    return q_x.float()  #由于pytorch不支持int类型的运算，所以还是得转成float表示

def quantize_tensor(x, scale, zero_point, num_bits, signed=True):
    qmin, qmax = get_quantize_limit(num_bits, signed)

    q_x = zero_point + x / scale
       
    q_x.clamp_(qmin, qmax).round_()

    return q_x.float()  #由于pytorch不支持int类型的运算，所以还是得转成float表示

def quantize_tensor_grad(x, scale, zero_point, e, epoch, num_bits, signed=True, loc=999):
    qmin, qmax = get_quantize_limit(num_bits, signed)
    
    # if scale == 0:
    #     # print('error in quantize_tensor, scale = 0')
    #     scale = 1e-6
    q_x = x / scale

    q_x.clamp_(qmin, qmax).round_()

    # mask1 = (scale/2 >= x) * (x > 0) 
    # mask2 = (-scale/2 <= x) * (x < 0) 
    # idx = torch.rand_like(q_x)
    # mask1 = (scale >= x) * (x > 0) * (idx < 0.2)
    # q_x[mask1] = 1
    # mask2 = (-scale <= x) * (x < 0) * (idx < 0.2)
    # q_x[mask2] = -1

    # mask1 = (scale/2 >= x) * (x > 0)
    # mask2 = (-scale/2 <= x) * (x < 0)
    # ms = (mask1 + mask2).sum()
    # nx = x.nelement()
    # if  ms/nx  > 0.2:
    #     # print('\n', loc, ms/nx)
    #     idx = torch.rand_like(x)
    #     mask = idx < 0.1*ms/nx
    #     q_x[mask1 * mask] = 1.
    #     q_x[mask2 * mask] = -1.


    q_x = q_x + zero_point
    return q_x.float()  #由于pytorch不支持int类型的运算，所以还是得转成float表示


def dequantize_tensor_W(q_x, scale, zero_point):
    # print(scale.shape)
    if len(q_x.shape) == 4:
        out = scale.view(-1, 1, 1, 1) * (q_x - zero_point.view(-1, 1, 1, 1))
    elif len(q_x.shape) == 2:
        out = scale.view(-1, 1) * (q_x - zero_point.view(-1, 1))

    if out.isnan().sum() != 0 or out.isinf().sum() != 0:
        print('test')
    return out

def dequantize_tensor(q_x, scale, zero_point):
    return scale * (q_x - zero_point)


def quantize_tensor_segment(grad, gbit, sc1, sc2, threshold):
    q_grad = torch.zeros_like(grad)

    q = 2 ** (gbit-2)
    
    mask1 = (-threshold <= grad) * (grad <= threshold)
    mask2 = grad < -threshold
    mask3 = grad > threshold

    try:
        q_grad[mask1] = (grad[mask1] / sc1).round()
    except ZeroDivisionError:
        print("sc1 is equal 0.")
    
    try:
        q_grad[mask2] = ((grad[mask2] + threshold)/sc2).round() - q
        q_grad[mask3] = ((grad[mask3] - threshold)/sc2).round() + q
    except ZeroDivisionError:
        print("sc2 is equal 0.")
    return q_grad, [mask1, mask2, mask3]

def dequantize_tensor_segment(q_grad, gbit, sc1, sc2, mask, threshold):
    f_grad = torch.zeros_like(q_grad)
    q = 2 ** (gbit-2)
    mask1, mask2, mask3 = mask
    f_grad[mask1] = q_grad[mask1] * sc1
    f_grad[mask2] = (q_grad[mask2] + q) * sc2 - threshold
    f_grad[mask3] = (q_grad[mask3] - q) * sc2 + threshold
    return f_grad

class FakeQuantize(Function):
    @staticmethod
    def forward(ctx, x, qparam):
        x = qparam.quantize_tensor(x)
        x = qparam.dequantize_tensor(x)
        return x
    
    @staticmethod
    def backward(ctx, grad_output):
        # print(torch.sum(grad_output))
        return grad_output, None

class STE(Function):
    @staticmethod
    def forward(ctx, x):
        x = torch.round(x)
        return x
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output





class QParam_W(nn.Module):
    # 这个类用于记录某一层的信息
    def __init__(self, w_bits, signed=True):
        super(QParam_W, self).__init__()
        self.w_bits = w_bits
        scale = torch.tensor([], requires_grad=False)
        zero_point = torch.tensor([], requires_grad=False)
        min = torch.tensor([], requires_grad=False)
        max = torch.tensor([], requires_grad=False)
        self.register_buffer('scale', scale)
        self.register_buffer('zero_point', zero_point)
        self.register_buffer('min', min)
        self.register_buffer('max', max)
        self.signed = signed

    def update(self, tensor):

        self.max, _ = tensor.contiguous().view(tensor.shape[0], -1).max(dim=1)
        self.min, _ = tensor.contiguous().view(tensor.shape[0], -1).min(dim=1)

        # if self.max.isnan().sum() != 0 or self.max.isinf().sum() != 0:
        #     print('test1', self.max.isnan().sum(), self.max.isinf().sum(), self.max)
        # if self.min.isnan().sum() != 0 or self.min.isinf().sum() != 0:
        #     print('test2', self.min.isnan().sum(), self.min.isinf().sum())

        # if self.scale.isnan().sum() != 0 or self.scale.isinf().sum():
        #     print('test3', self.scale.isnan().sum(), self.scale.isinf().sum())

        self.scale, self.zero_point = calcScaleZeroPoint_W(self.min, self.max, self.w_bits, signed=self.signed, mode=self.mode)
        # print(self.scale.shape, self.zero_point.shape)

        # if self.scale.isnan().sum() != 0 or self.scale.isinf().sum():
        #     print('test4', self.scale.isnan().sum(), self.scale.isinf().sum())

    def quantize_tensor(self, tensor):
        return quantize_tensor_W(tensor, self.scale, self.zero_point, num_bits=self.w_bits, signed=self.signed)
    
    def dequantize_tensor(self, q_x):
        return dequantize_tensor_W(q_x, self.scale, self.zero_point)
        
    def update_quantize_parameters(self, module, M, qi, qo):
        tmax = module.weight.view(module.weight.shape[0], -1).abs().max(dim=1).values
        mask = tmax.round() >= 2 ** (self.w_bits - 1)
        nmask = ~mask
        weight = module.weight.clone()
        bias = module.bias.clone()
        # print(weight.shape)

        # if module.weight.isnan().sum() != 0 or module.weight.isinf().sum():
        #     print('test')

        wmax = self.max.clone()
        wmin = self.min.clone()
        wscale = self.scale.clone()
        wzero = self.zero_point.clone()
        M1 = M.view(-1).clone()

        # if self.scale.isnan().sum() != 0 or self.scale.isinf().sum():
        #     print('test')

        fp_weight = self.dequantize_tensor(module.weight.data.round())
        fp_bias = dequantize_tensor(module.bias.data.round(), scale=qi.scale * self.scale, zero_point=0.)
        
        # if fp_weight.isnan().sum() != 0 or fp_weight.isinf().sum():
        #     print('test')

        self.update(fp_weight)
        
        M.data = (self.scale * qi.scale / qo.scale).data.view(-1)

        if len(module.weight.shape) == 4:
            self.zero_point = self.zero_point.view(-1, 1, 1, 1)
        elif len(module.weight.shape) == 2:
            self.zero_point = self.zero_point.view(-1, 1)
        # print(self.qw.zero_point.shape)

        self.max[nmask] = wmax[nmask]
        self.min[nmask] = wmin[nmask]
        self.scale[nmask] = wscale[nmask]
        self.zero_point[nmask] = wzero[nmask]
        M[nmask] = M1[nmask]
        if len(module.weight.shape) == 4:
            M = M.view(1, -1, 1, 1)
        elif len(module.weight.shape) == 2:
            M = M.view(1, -1)

        # if module.weight.data.isnan().sum() != 0 or module.weight.data.isinf().sum() != 0:
        #     print('test')
 
        module.weight.data = self.quantize_tensor(fp_weight.data)
        # if module.weight.data.isnan().sum() != 0:
        #     print('test')
        module.bias.data = quantize_tensor(fp_bias, scale=qi.scale * self.scale,
                                            zero_point=0, num_bits=32, signed=True)
        module.weight.data[nmask] = weight[nmask].data
        module.bias.data[nmask] = bias[nmask].data

        if module.bias.data.abs().max().round() >= 2 ** 31:
            module.bias.data = module.bias.data.clamp(-2 ** 31 + 1, 2 **31 - 1)
        return M, module

class QParam(nn.Module):
    # 这个类用于记录某一层的信息
    def __init__(self, a_bits, signed=True, isWeight=False):
        super(QParam, self).__init__()
        self.a_bits = a_bits
        scale = torch.tensor([], requires_grad=False)
        zero_point = torch.tensor([], requires_grad=False)
        min = torch.tensor([], requires_grad=False)
        max = torch.tensor([], requires_grad=False)
        
        self.register_buffer('zero_point', zero_point)
        # if isWeight:
        #     self.register_parameter('scale', nn.Parameter(torch.tensor(1e-10), requires_grad=True))
        # else:
        self.register_buffer('scale', scale)    

        self.register_buffer('min', min)
        self.register_buffer('max', max)
        self.signed = signed
        self.isWeight = isWeight
        self.register_buffer('epoch_max', torch.tensor(-999, requires_grad=False))
        self.register_buffer('epoch_min', torch.tensor(999, requires_grad=False))
        self.mode = 'asymmetry'

    def update(self, tensor):
        if not self.isWeight:       # activate
            if self.max.nelement() == 0 or self.max.data < tensor.max().data:
                temp = self.max.data if self.max.nelement() != 0 else torch.tensor(0.0)
                self.max.data = tensor.max().data * 0.8 + temp * 0.2
            self.max.clamp_(min=0)

            if self.min.nelement() == 0 or self.min.data > tensor.min().data:
                temp = self.min.data if self.min.nelement() != 0 else torch.tensor(0.0)
                self.min.data = tensor.min().data * 0.8 + temp * 0.2
            self.min.clamp_(max=0)
        else:
            if self.mode == 'symmetry':
                self.max = tensor.abs().max()
                self.max.clamp_(min=0)

                self.min = -tensor.abs().max()
                self.min.clamp_(max=0)

            else:
                self.max.data = tensor.max().data
                self.max.clamp_(min=0)

                self.min.data = tensor.min().data
                self.min.clamp_(max=0)

        self.scale, self.zero_point = calcScaleZeroPoint(self.min, self.max, self.a_bits, signed=self.signed )

    def quantize_tensor(self, tensor):
        return quantize_tensor(tensor, self.scale, self.zero_point, num_bits=self.a_bits, signed=self.signed)
    
    def quantize_tensor_grad(self, tensor):
        return quantize_tensor_grad(tensor, self.scale, self.zero_point, num_bits=self.a_bits, signed=self.signed)
    
    def dequantize_tensor(self, q_x):
        return dequantize_tensor(q_x, self.scale, self.zero_point)
    
    def update_quantize_activate(self, qx):
        f_x = self.dequantize_tensor(qx)
        vmax = f_x.max()
        vmin = f_x.min()
        # if vmax > self.max and vmax / self.max < 2:
        #     self.max.data = self.max.data * 0.9 + vmax.data * 0.1
        # if vmin < self.min and self.min != 0 and vmin / self.min < 2:
        #     self.min.data = self.min.data * 0.9 + vmin.data * 0.1

        mask_max = f_x > self.max
        if mask_max.sum() / mask_max.nelement() >= 0.1 and vmax <= 1:
            self.max.data = self.max.data * 0.9 + vmax.data * 0.1
        
        if self.min != 0:
            mask_min = f_x < self.min
            if mask_min.sum() / mask_min.nelement() >= 0.1 and vmin >= -1:
                self.min.data = self.min.data * 0.9 + vmin.data * 0.1

        self.scale, self.zero_point = calcScaleZeroPoint(self.min, self.max, self.a_bits, signed=self.signed)

    def record_activate_value(self, qx):
        f_x = self.dequantize_tensor(qx)
        e_max = f_x.max()
        e_min = f_x.min()
        if self.epoch_max < e_max and e_max <= 100:
            if self.epoch_max == -999:
                self.epoch_max = e_max
            else:
                self.epoch_max = self.epoch_max * 0.8 + e_max * 0.2
        if self.epoch_min > e_min and self.min != 0 and e_min >= -100:
            if self.epoch_min == 999:
                self.epoch_min = e_min
            else:
                self.epoch_min = self.epoch_min * 0.8 + e_min * 0.2

    def set_activate_value(self):
        self.max.data = self.max.data if self.epoch_max.data == -999 else self.epoch_max.data
        self.min.data = self.min.data if self.epoch_min.data == 999 else self.epoch_min.data
        self.scale, self.zero_point = calcScaleZeroPoint(self.min, self.max, self.a_bits, signed=self.signed)
        self.epoch_max = torch.tensor(-999)
        self.epoch_min = torch.tensor(999)
    
    def update_quantize_parameters(self, module, M, qi, qo):
        qmin, qmax = get_quantize_limit(self.a_bits, self.signed)
        if module.weight.round().max() > qmax or \
            module.weight.round().max() < qmax or \
                module.weight.round().min() > qmin or\
                      module.weight.round().min() < qmin:
            fp_weight = self.dequantize_tensor(module.weight.data)
            fp_bias = dequantize_tensor(module.bias.data, scale=qi.scale * self.scale, zero_point=0.)
            
            # if fp_weight.isnan().sum() != 0 or fp_weight.isinf().sum():
            #     print('test')

            self.update(fp_weight)
            
            M.data = (self.scale * qi.scale / qo.scale).data

            module.weight.data = self.quantize_tensor(fp_weight.data)
            module.bias.data = quantize_tensor(fp_bias, scale=qi.scale * self.scale,
                                            zero_point=0, num_bits=32, signed=True)

        # if module.bias.data.abs().max().round() >= 2 ** 31:
        module.bias.data = module.bias.data.clamp(-2 ** 31 + 1, 2 **31 - 1)

        # module.weight.data=module.weight.data.round()
        # module.bias.data=module.bias.data.round()
        return M, module

class QModule(nn.Module):
    # 用qi和qo来记录当前层的信息，包括最大最小值，缩放因子，零点
    def __init__(self, a_bits, qi=True, qo=True, signed=False):
        super(QModule, self).__init__()
        if qi:
            self.qi = QParam(a_bits=a_bits, signed=signed)
        if qo:
            self.qo = QParam(a_bits=a_bits, signed=signed)
        
    def freeze(self):
        pass

    def quantize_inference(self, x):
        raise NotImplementedError('quantize_inference should be implemented.')