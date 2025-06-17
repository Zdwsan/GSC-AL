import torch
import torch.nn.functional as F
import yaml
from models.quantize_dequantize import calcScaleZeroPoint, quantize_tensor, dequantize_tensor
from models.quantize_dequantize import get_quantize_limit


def update_parameter():
    with open('config/config.yaml', 'r') as file:
        config = yaml.safe_load(file)
    global method, gbit
    method = config['method']
    gbit = config['gbit']

class _QTruncate(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, num_bits, signed):
        ctx.num_bits = num_bits
        x = x.round()

        qmin, qmax = get_quantize_limit(num_bits, signed)

        mask1 = x >= qmin
        mask2 = x <= qmax
        x = x.clamp_(qmin, qmax).round_()

        ctx.save_for_backward(mask1, mask2)
        return x
    
    @staticmethod
    def backward(ctx, grad_output):
        mask1, mask2 = ctx.saved_tensors
        mask = mask1 * mask2
        grad_x = grad_output * mask
        return grad_x, None, None, None

class _QConv2dFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias, stride, padding, dilation, groups, qi, qo, qw_scale, qw_zero, M, signed, w_bits):
        x = x.round()
        weight = weight.round()
        bias = bias.round()

        lower, upper = get_quantize_limit(w_bits, signed)
        weight = weight.clamp(lower, upper)
        # if signed:
        #     weight = weight.clamp(- 2. ** (w_bits - 1) + 1, 2. ** (w_bits - 1) - 1)
        # else:
        #     weight = weight.clamp(0., 2. ** w_bits - 1)
        bias = bias.clamp(-2**31+1, 2**31-1)

        # print('======================================')
        x = x - qi.zero_point  # ensure x is int
        weight = weight - qw_zero  # ensure weight is int

        ctx.stride = stride
        ctx.padding = padding
        ctx.dilation = dilation
        ctx.groups = groups
        ctx.input_size = x.shape
        ctx.weight_size = weight.shape

        # M = (qi.scale * qw.scale / qo.scale).view(1, -1, 1, 1)        

        out = F.conv2d(x, weight, bias, stride, padding, dilation, groups)
        out = out * M
        out = out.round()
        out = (out + qo.zero_point)
        # print(out.min(), out.max())
        ctx.save_for_backward(weight, M, x)
        # print(123)
        return out

    @staticmethod
    def backward(ctx, grad_output):



        weight, M, x = ctx.saved_tensors

        _grad_output = grad_output * M.view(1, -1, 1, 1)    # sw * si / so, sw 大， so大， 

        grad_bias = _grad_output.sum([0, 2, 3]) #* M.view(-1)

        maxv = _grad_output.max()
        minv = _grad_output.min()
        scale, zero_point = calcScaleZeroPoint(minv, maxv, num_bits=gbit, signed=True )

        q_grad_output = quantize_tensor(_grad_output, scale, zero_point, num_bits=gbit, signed=True) - zero_point


        grad_x = torch.nn.grad.conv2d_input(ctx.input_size, weight, q_grad_output, 
                                            stride=ctx.stride, padding=ctx.padding,
                                            dilation=ctx.dilation, groups=ctx.groups) #* scale * wscale

        
        grad_w = torch.nn.grad.conv2d_weight(x, ctx.weight_size, q_grad_output,
                                            stride=ctx.stride, padding=ctx.padding,
                                            dilation=ctx.dilation, groups=ctx.groups) #* scale * iscale

        grad_x = dequantize_tensor(grad_x, scale=scale, zero_point=0)

        if method != 'GSC_AL':
            grad_w = dequantize_tensor(grad_w, scale=scale, zero_point=0)
        
        return grad_x, grad_w, grad_bias, None, None, None, None, None, None, None, None, None, None, None, None, None, None

class _QLinearFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias, qi, qo, qw_scale, qw_zero, M, signed, w_bits):
        x = x.round()
        weight = weight.round()
        bias = bias.round()
        
        lower, upper = get_quantize_limit(w_bits, signed)
        weight = weight.clamp(lower, upper)

        bias = bias.clamp(-2**31+1, 2**31-1)

        x = x - qi.zero_point  # ensure x is int
        weight = weight - qw_zero  # ensure weight is int

        ctx.input_size = x.shape
        ctx.weight_size = weight.shape

        out = F.linear(x, weight, bias)
        out = out * M
        out = out.round()
        out = out + qo.zero_point

        ctx.save_for_backward(weight, M, x, out, qw_zero, qi.zero_point, qo.zero_point, qw_scale, qi.scale, qo.scale)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        weight, M, x, out, wzero, izero, ozero, wscale, iscale, oscale = ctx.saved_tensors

        _grad_output = grad_output * M.view(1, -1)
        
        grad_bias = _grad_output.sum(0) #* M.view(-1)

        maxv = _grad_output.max()
        minv = _grad_output.min()
        scale, zero_point = calcScaleZeroPoint(minv, maxv, num_bits=gbit, signed=True )

        q_grad_output = quantize_tensor(_grad_output, scale, zero_point, num_bits=gbit, signed=True) - zero_point

        grad_x = torch.matmul(q_grad_output, weight) 

        grad_w = torch.matmul(x.T, q_grad_output).T
        grad_x = dequantize_tensor(grad_x, scale=scale, zero_point=0.)

        if method != 'GSC_AL':
            grad_w = dequantize_tensor(grad_w, scale=scale, zero_point=0)

        return grad_x, grad_w, grad_bias, None, None, None, None, None, None, None, None, None, None


class _QAddFunct(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x1, x2, qi1, qi2, qo):
        x1 = x1.round()
        x2 = x2.round()
        M1 = qi1.scale / qo.scale
        M2 = qi2.scale / qo.scale
        ctx.M1 = M1
        ctx.M2 = M2
        x = (x1 - qi1.zero_point) * M1 + \
            (x2 - qi2.zero_point) * M2
        x.round_()
        x = x + qo.zero_point
        return x

    @staticmethod
    def backward(ctx, grad_output):
        grad_x1 = grad_output * ctx.M1
        grad_x2 = grad_output * ctx.M2
        return grad_x1, grad_x2, None, None, None


class _QAvgpoolFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, kernel_size, stride, padding, qi, qo):
        x = x.round()
        
        ctx.input_shape = x.shape

        x = x - qi.zero_point
        x = F.avg_pool2d(x, kernel_size, stride, padding)
        M = qi.scale / qo.scale
        ctx.M = M
        x = x * M
        x.round_()
        x = x + qo.zero_point
        return x

    @staticmethod
    def backward(ctx, grad_output):
        input_shape = ctx.input_shape
        grad_input = grad_output.repeat(1, 1, *input_shape[-2:]) / (input_shape[-1] * input_shape[-2]) * ctx.M
        return grad_input, None, None, None, None, None
    

class _QMaxpoolFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, kernel_size, stride, padding, qi, qo):
        x = x.round()
        ctx.input_shape = x.shape
        x = x - qi.zero_point
        x, indices = F.max_pool2d_with_indices(x, kernel_size, stride, padding)
        ctx.save_for_backward(indices)
        x = x * qi.scale / qo.scale
        x.round_()
        x = x + qo.zero_point
        return x

    @staticmethod
    def backward(ctx, grad_output):
        batch, out_channels, w, h = ctx.input_shape

        indices, = ctx.saved_tensors
        grad = torch.zeros((batch, out_channels, w*h))
        indices_ = indices.view(batch, out_channels, -1)
        grad_ = grad_output.view(batch, out_channels, -1)
        for i in range(batch):
            for j in range(out_channels):
                for k in range(grad_.shape[-1]):
                    grad[i, j, indices_[i, j, k]] = grad_[i, j, k]
        grad_input = grad.view(batch, out_channels, w, h)

        return grad_input, None, None, None, None, None











