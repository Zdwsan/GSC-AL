

from models.quantize_resnet18 import QuantizeResnet18
from models.quantize_class import *

def get_net_parameters(model, plist):
    for n, m in model.named_children():
        # print(n)
        if isinstance(m, QConvBNReLU):
            if m.conv_module.weight is not None and m.conv_module.kernel_size[0] > 0:
                plist.append(m.conv_module.weight)
            if m.conv_module.bias is not None and m.conv_module.kernel_size[0] > 0:
                plist.append(m.conv_module.bias)

        elif isinstance(m, QLinear):
            if m.fc_module.weight is not None:
                plist.append(m.fc_module.weight)
            if m.fc_module.bias is not None:
                plist.append(m.fc_module.bias)

        elif isinstance(m, (nn.Module, nn.ModuleList, nn.Sequential)):
            plist = get_net_parameters(m, plist)
    return plist

def getModel(num_class, modelname, signed, w_bits, a_bits):
    if modelname == 'resnet18':
        model = QuantizeResnet18(num_class, signed=signed, w_bits=w_bits, a_bits=a_bits)

    return model
    