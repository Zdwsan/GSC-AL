import torch.nn as nn

from models.quantize_class import QConvBNReLU, QLinear
from models.quantize_resnet18 import QuantizeResnet18


def get_net_parameters(model, parameters):
    for _, module in model.named_children():
        if isinstance(module, QConvBNReLU):
            if (
                module.conv_module.weight is not None
                and module.conv_module.kernel_size[0] > 0
            ):
                parameters.append(module.conv_module.weight)
            if (
                module.conv_module.bias is not None
                and module.conv_module.kernel_size[0] > 0
            ):
                parameters.append(module.conv_module.bias)

        elif isinstance(module, QLinear):
            if module.fc_module.weight is not None:
                parameters.append(module.fc_module.weight)
            if module.fc_module.bias is not None:
                parameters.append(module.fc_module.bias)

        elif isinstance(module, (nn.Module, nn.ModuleList, nn.Sequential)):
            get_net_parameters(module, parameters)
    return parameters


def getModel(num_class, modelname, signed, w_bits, a_bits):
    if modelname == "resnet18":
        return QuantizeResnet18(
            num_class,
            signed=signed,
            w_bits=w_bits,
            a_bits=a_bits,
        )

    raise ValueError(f"Unsupported model: {modelname}")
