import torch
import torch.nn as nn
from models.quantize_class import QConvBNReLU, QLinear
import yaml

global maxGrad_conv, maxGrad_fc
def get_maxGrad():
    with open('config/config.yaml', 'r') as file:
        config = yaml.safe_load(file)
    global maxGrad_conv, maxGrad_fc
    maxGrad_conv = config['maxGrad_conv']
    maxGrad_fc = config['maxGrad_fc']

class Adamwscale_2(torch.optim.AdamW):
    @staticmethod
    def pre_step(model, e, epoch, tlayer):
        for n, m in model.named_children():
            if isinstance(m, QConvBNReLU):
                if m.conv_module.bias.grad is not None and m.conv_module.kernel_size[0] > 0:
                    tscale = (m.qw.scale * m.qi.scale) ** 2
                    m.conv_module.bias.grad.data = m.conv_module.bias.grad.data / tscale  

                if m.conv_module.weight.grad is not None and m.conv_module.kernel_size[0] > 0:
                    c_max = m.conv_module.weight.grad.abs().contiguous().view(m.conv_module.weight.grad.shape[0], -1).max(dim=1).values # 按通道来
                    mask0 = c_max == 0
                    c_max[mask0] = 1e-10
                    if m.bl < int(tlayer*0.33):
                        tscale3 = maxGrad_conv / c_max
                    elif int(tlayer*0.33) <= m.bl < int(tlayer*0.66):
                        tscale3 = (maxGrad_conv/2) / c_max
                    else:
                        tscale3 = (maxGrad_conv/4) / c_max
                    m.conv_module.weight.grad.data = m.conv_module.weight.grad.data * tscale3.view(-1, 1, 1, 1)

            if isinstance(m, QLinear):
                if m.fc_module.bias.grad is not None:
                    tscale = (m.qw.scale * m.qi.scale) ** 2
                    m.fc_module.bias.grad.data = m.fc_module.bias.grad.data / tscale  

                if m.fc_module.weight.grad is not None:
                    c_max = m.fc_module.weight.grad.abs().max().values
                    mask0 = c_max == 0
                    c_max[mask0] = 1e-10
                    tscale3 = maxGrad_fc / c_max * (1 - (e/epoch)*1) 
                    m.fc_module.weight.grad.data = m.fc_module.weight.grad.data * tscale3.view(-1, 1)  

            elif isinstance(m, (nn.Module, nn.ModuleList, nn.Sequential)):
                Adamwscale_2.pre_step(m, e, epoch, tlayer)


class AdamwScale(torch.optim.AdamW):
    @staticmethod
    def pre_step(model):
        for n, m in model.named_children():
            if isinstance(m, QConvBNReLU):
                if m.conv_module.bias.grad is not None:
                    tscale = (m.qw.scale * m.qi.scale) ** 2
                    m.conv_module.bias.grad.data = m.conv_module.bias.grad.data / tscale  

                if m.conv_module.weight.grad is not None:
                    tscale = m.qw.scale ** 2
                    m.conv_module.weight.grad.data = m.conv_module.weight.grad.data / tscale.view(-1, 1, 1, 1)  
                    m.conv_module.weight.grad.data = m.conv_module.weight.grad.data.clamp(-500, 500)
            elif isinstance(m, QLinear):
                if m.fc_module.bias.grad is not None:
                    tscale = (m.qw.scale * m.qi.scale) ** 2
                    m.fc_module.bias.grad.data = m.fc_module.bias.grad.data / tscale

                if m.fc_module.weight.grad is not None:
                    tscale = m.qw.scale ** 2
                    m.fc_module.weight.grad.data = m.fc_module.weight.grad.data / tscale.view(-1, 1)

            elif isinstance(m, (nn.Module, nn.ModuleList, nn.Sequential)):
                AdamwScale.pre_step(m)

# def update_lr( epoch, lr_master_S,  optimizer_S):
#     """
#     update learning rate of optimizers
#     :param epoch: current training epoch
#     """
#     lr_S = lr_master_S.get_lr(epoch)

#     # update learning rate of model optimizer
#     for param_group in optimizer_S.param_groups:
#         param_group['lr'] = lr_S
#         # print(param_group['params'])
#     return lr_S

