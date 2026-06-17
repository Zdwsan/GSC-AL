import torch
import torch.nn as nn
from torch.autograd import Function


def get_quantize_limit(bits, signed):
    if signed:
        upper = 2 ** (bits - 1) - 1
        lower = -(2 ** (bits - 1)) + 1
    elif signed == False:
        upper = 2**bits - 1
        lower = 0.0
    return lower, upper


def calcScaleZeroPoint(min_val, max_val, num_bits, signed=True):
    qmin, qmax = get_quantize_limit(num_bits, signed)
    scale = (max_val - min_val) / (qmax - qmin)

    if scale == 0:
        scale = torch.tensor(1e-4).to(min_val.device)
        zero_point = torch.tensor(0.0).to(min_val.device)
    else:
        zero_point = qmax - max_val / scale

    if zero_point < qmin:
        zero_point = torch.tensor([qmin], dtype=torch.float32).to(min_val.device)
    elif zero_point > qmax:
        zero_point = torch.tensor([qmax], dtype=torch.float32).to(max_val.device)

    zero_point.round_()

    return scale, zero_point


def quantize_tensor(x, scale, zero_point, num_bits, signed=True):
    qmin, qmax = get_quantize_limit(num_bits, signed)

    q_x = zero_point + x / scale

    q_x.clamp_(qmin, qmax).round_()

    return q_x.float()


def dequantize_tensor(q_x, scale, zero_point):
    return scale * (q_x - zero_point)


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


class QParam(nn.Module):
    def __init__(self, a_bits, signed=True, isWeight=False):
        super(QParam, self).__init__()
        self.a_bits = a_bits
        scale = torch.tensor([], requires_grad=False)
        zero_point = torch.tensor([], requires_grad=False)
        min = torch.tensor([], requires_grad=False)
        max = torch.tensor([], requires_grad=False)

        self.register_buffer("zero_point", zero_point)
        self.register_buffer("scale", scale)

        self.register_buffer("min", min)
        self.register_buffer("max", max)
        self.signed = signed
        self.isWeight = isWeight
        self.register_buffer("epoch_max", torch.tensor(-999, requires_grad=False))
        self.register_buffer("epoch_min", torch.tensor(999, requires_grad=False))
        self.mode = "asymmetry"

    def update(self, tensor):
        if not self.isWeight:  # activate
            if self.max.nelement() == 0 or self.max.data < tensor.max().data:
                temp = self.max.data if self.max.nelement() != 0 else torch.tensor(0.0)
                self.max.data = tensor.max().data * 0.8 + temp * 0.2
            self.max.clamp_(min=0)

            if self.min.nelement() == 0 or self.min.data > tensor.min().data:
                temp = self.min.data if self.min.nelement() != 0 else torch.tensor(0.0)
                self.min.data = tensor.min().data * 0.8 + temp * 0.2
            self.min.clamp_(max=0)
        else:
            if self.mode == "symmetry":
                self.max = tensor.abs().max()
                self.max.clamp_(min=0)

                self.min = -tensor.abs().max()
                self.min.clamp_(max=0)

            else:
                self.max.data = tensor.max().data
                self.max.clamp_(min=0)

                self.min.data = tensor.min().data
                self.min.clamp_(max=0)

        self.scale, self.zero_point = calcScaleZeroPoint(
            self.min, self.max, self.a_bits, signed=self.signed
        )

    def quantize_tensor(self, tensor):
        return quantize_tensor(
            tensor,
            self.scale,
            self.zero_point,
            num_bits=self.a_bits,
            signed=self.signed,
        )

    def dequantize_tensor(self, q_x):
        return dequantize_tensor(q_x, self.scale, self.zero_point)

    def update_quantize_activate(self, qx):
        f_x = self.dequantize_tensor(qx)
        vmax = f_x.max()
        vmin = f_x.min()

        mask_max = f_x > self.max
        if mask_max.sum() / mask_max.nelement() >= 0.1 and vmax <= 1:
            self.max.data = self.max.data * 0.9 + vmax.data * 0.1

        if self.min != 0:
            mask_min = f_x < self.min
            if mask_min.sum() / mask_min.nelement() >= 0.1 and vmin >= -1:
                self.min.data = self.min.data * 0.9 + vmin.data * 0.1

        self.scale, self.zero_point = calcScaleZeroPoint(
            self.min, self.max, self.a_bits, signed=self.signed
        )

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
        self.max.data = (
            self.max.data if self.epoch_max.data == -999 else self.epoch_max.data
        )
        self.min.data = (
            self.min.data if self.epoch_min.data == 999 else self.epoch_min.data
        )
        self.scale, self.zero_point = calcScaleZeroPoint(
            self.min, self.max, self.a_bits, signed=self.signed
        )
        self.epoch_max = torch.tensor(-999)
        self.epoch_min = torch.tensor(999)

    def update_quantize_parameters(self, module, M, qi, qo):
        qmin, qmax = get_quantize_limit(self.a_bits, self.signed)
        if (
            module.weight.round().max() > qmax
            or module.weight.round().max() < qmax
            or module.weight.round().min() > qmin
            or module.weight.round().min() < qmin
        ):
            fp_weight = self.dequantize_tensor(module.weight.data)
            fp_bias = dequantize_tensor(
                module.bias.data, scale=qi.scale * self.scale, zero_point=0.0
            )

            self.update(fp_weight)

            M.data = (self.scale * qi.scale / qo.scale).data

            module.weight.data = self.quantize_tensor(fp_weight.data)
            module.bias.data = quantize_tensor(
                fp_bias,
                scale=qi.scale * self.scale,
                zero_point=0,
                num_bits=32,
                signed=True,
            )

        module.bias.data = module.bias.data.clamp(-(2**31) + 1, 2**31 - 1)
        return M, module


class QModule(nn.Module):
    def __init__(self, a_bits, qi=True, qo=True, signed=False):
        super(QModule, self).__init__()
        if qi:
            self.qi = QParam(a_bits=a_bits, signed=signed)
        if qo:
            self.qo = QParam(a_bits=a_bits, signed=signed)

    def freeze(self):
        pass

    def quantize_inference(self, x):
        raise NotImplementedError("quantize_inference should be implemented.")
