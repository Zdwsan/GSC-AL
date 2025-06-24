import torch.nn as nn

import torch
from models.quantize_dequantize import *
from models.quantize_class import *

from models.fpn import *


class QuantizeResidual(nn.Module):
    def __init__(self, residual, w_bits, a_bits, signed, bl):
        super(QuantizeResidual, self).__init__()
        self.residual = residual
        self.w_bits = w_bits
        self.a_bits = a_bits
        self.signed = signed
        self.ineqout = residual.ineqout
        self.bl = bl
    def forward(self, x):
        out = self.residual.layer3(self.residual.layer2(self.residual.layer1(x)))      
        if self.ineqout:
            out = out + x
        else:
            out = self.residual.shortcut(x) + out
        return 

    def quantize(self, training=True):
        self.qlayer1 = QConvBNReLU(self.residual.layer1.conv, bn_module=self.residual.layer1.bn, qi=False, qo=True, w_bits=self.w_bits, a_bits=self.a_bits, 
                                   activate=self.residual.layer1.activate, signed=self.signed, bl=self.bl)
        self.qlayer2 = QConvBNReLU(self.residual.layer2.conv, bn_module=self.residual.layer2.bn, qi=False, qo=True, w_bits=self.w_bits, a_bits=self.a_bits, 
                                   activate=self.residual.layer2.activate, signed=self.signed, bl=self.bl)
        self.qlayer3 = QConvBNReLU(self.residual.layer3.conv, bn_module=self.residual.layer3.bn, qi=False, qo=True, w_bits=self.w_bits, a_bits=self.a_bits, 
                                   activate=self.residual.layer3.activate, signed=self.signed, bl=self.bl)
        self.qadd = QAddReLU(qi1=False, qi2=False, qo=True, a_bits=self.a_bits, signed=self.signed, activate=hasattr(self.residual, 'relu'))
        
        if self.ineqout is not True:
            self.qshortcut = QConvBNReLU(self.residual.shortcut.conv, bn_module=self.residual.shortcut.bn, qi=False, qo=True, w_bits=self.w_bits, a_bits=self.a_bits, 
                                  activate=self.residual.shortcut.activate, signed=self.signed, bl=self.bl)

    def freeze(self, pre_qo):
        self.qlayer1.freeze(qi=pre_qo)
        self.qlayer2.freeze(qi=self.qlayer1.qo)
        self.qlayer3.freeze(qi=self.qlayer2.qo)
        if self.ineqout:
            self.qadd.freeze(qi1=pre_qo, qi2=self.qlayer3.qo)
        else:
            self.qshortcut.freeze(qi=pre_qo)
            self.qadd.freeze(qi1=self.qshortcut.qo, qi2=self.qlayer3.qo)

    def get_last_layer_qo(self):
        # return self.qlayer3.qo
        return self.qadd.qo
 
    def quantize_forward(self, x):
        out = self.qlayer1(x)
        
        out = self.qlayer2(out)  
        
        out = self.qlayer3(out)
        
        if self.ineqout:
            out = self.qadd(x, out) 
        else:
            s = self.qshortcut(x)
            out = self.qadd(s, out) 
        return out
 
    def quantize_inference(self, x):
        qx = self.qlayer1.quantize_inference(x)
        qx = self.qlayer2.quantize_inference(qx)
        qx = self.qlayer3.quantize_inference(qx)
        if self.ineqout:
            qx = self.qadd.quantize_inference(x, qx)
        else:
            qs = self.qshortcut.quantize_inference(x)
            qx = self.qadd.quantize_inference(qs, qx)
        
        return qx

    def quantize_training(self, x, record_act):

        out = self.qlayer1.quantize_training(x)
        record_act.append(out - self.qlayer1.qo.zero_point)

        out = self.qlayer2.quantize_training(out)
        record_act.append(out - self.qlayer2.qo.zero_point)

        out = self.qlayer3.quantize_training(out)
        record_act.append(out - self.qlayer3.qo.zero_point)

        if self.ineqout:
            out = self.qadd.quantize_training(x, out)
        else:
            qs = self.qshortcut.quantize_training(x)
            record_act.append(qs - self.qshortcut.qo.zero_point)

            out = self.qadd.quantize_training(qs, out)
        return out

    def update_quantize_parameters(self):
        self.qlayer1.update_quantize_parameters()
        self.qlayer2.update_quantize_parameters()
        self.qlayer3.update_quantize_parameters()
        if self.ineqout is not True:
            self.qshortcut.update_quantize_parameters()
    


class QuantizeInvertedResidual_resnet18(nn.Module):
    def __init__(self, residual, w_bits, a_bits, signed, bl):
        super(QuantizeInvertedResidual_resnet18, self).__init__()
        self.residual = residual
        self.w_bits = w_bits
        self.a_bits = a_bits 
        self.signed = signed
        self.bl = bl
    def quantize(self):
        self.qlayer1 = QConvBNReLU(self.residual.layer1.conv, self.residual.layer1.bn, qi=False, qo=True, w_bits=self.w_bits, a_bits=self.a_bits, activate=self.residual.layer1.activate, signed=self.signed, bl=self.bl)
        self.qlayer2 = QConvBNReLU(self.residual.layer2.conv, self.residual.layer2.bn, qi=False, qo=True, w_bits=self.w_bits, a_bits=self.a_bits, activate=self.residual.layer2.activate, signed=self.signed, bl=self.bl)
        if self.residual.downsample:
            self.qshortcut = QConvBNReLU(self.residual.shortcut.conv, self.residual.shortcut.bn, qi=False, qo=True, w_bits=self.w_bits, a_bits=self.a_bits, activate=self.residual.shortcut.activate, signed=self.signed, bl=self.bl)
        self.qadd = QAddReLU(qi1=False, qi2=False, qo=True, a_bits=self.a_bits, signed=self.signed, activate=hasattr(self.residual, 'relu'))

    def freeze(self, pre_qo):
        self.qlayer1.freeze(qi=pre_qo)
        self.qlayer2.freeze(qi=self.qlayer1.qo)
        if self.residual.downsample:
            self.qshortcut.freeze(qi=pre_qo)
            self.qadd.freeze(qi1=self.qshortcut.qo, qi2=self.qlayer2.qo)
        else:
            self.qadd.freeze(qi1=pre_qo, qi2=self.qlayer2.qo)

    def update_quantize_parameters(self):
        self.qlayer1.update_quantize_parameters()
        self.qlayer2.update_quantize_parameters()
        if self.residual.downsample:
            self.qshortcut.update_quantize_parameters()

    def get_last_layer_qo(self):
        return self.qadd.qo

    def quantize_forward(self, x):
        out = self.qlayer1(x)
        out = self.qlayer2(out)        
        if self.residual.downsample:
            out2 = self.qshortcut(x)
            out = self.qadd(out2, out)    
        else:
            out = self.qadd(x, out)
        return out

    def quantize_inference(self, x):

        qx = self.qlayer1.quantize_inference(x)
        qx = self.qlayer2.quantize_inference(qx)

        if self.residual.downsample:
            qx2 = self.qshortcut.quantize_inference(x)
            qx = self.qadd.quantize_inference(qx2, qx)
        else: 
            qx = self.qadd.quantize_inference(x, qx)
        return qx
    
    def quantize_training(self, x, record_act):
        qx = self.qlayer1.quantize_training(x)
        record_act.append(qx - self.qlayer1.qo.zero_point)

        qx = self.qlayer2.quantize_training(qx)
        record_act.append(qx - self.qlayer2.qo.zero_point)

        if self.residual.downsample:
            qx2 = self.qshortcut.quantize_training(x)
            record_act.append(qx2 - self.qshortcut.qo.zero_point)
            qx = self.qadd.quantize_training(qx2, qx)
        else:
            qx = self.qadd.quantize_training(x, qx)
        return qx


class QuantizeResnet18(Resnet18_FPN):
    def __init__(self, num_class, signed=True, w_bits=8, a_bits=8):
        super(QuantizeResnet18, self).__init__(num_classes=num_class, img_size=448, strides=[8, 16, 32])
        self.signed = signed
        self.w_bits = w_bits
        self.a_bits = a_bits

    def quantize(self):
        self.qadd4 = QAddReLU(a_bits=self.a_bits, qi1=False, qi2=False, qo=True, signed=self.signed, activate=False)
        self.qadd3 = QAddReLU(a_bits=self.a_bits, qi1=False, qi2=False, qo=True, signed=self.signed, activate=False)


        self.qneck5 = QuantizeResidual(self.neck5, w_bits=self.w_bits, a_bits=self.a_bits, signed=self.signed, bl=0)
        self.qneck5.quantize()

        self.qneck4 = QuantizeResidual(self.neck4, w_bits=self.w_bits, a_bits=self.a_bits, signed=self.signed, bl=0)
        self.qneck4.quantize()

        self.qneck3 = QuantizeResidual(self.neck3, w_bits=self.w_bits, a_bits=self.a_bits, signed=self.signed, bl=0)
        self.qneck3.quantize()

        bl = 1
        self.qfeature = [0 for _ in range(len(self.net.feature))]
        for i in range(len(self.net.feature)):
            self.qfeature[-1-i] = QuantizeInvertedResidual_resnet18(self.net.feature[-1-i], w_bits=self.w_bits, a_bits=self.a_bits, signed=self.signed, bl=bl)
            self.qfeature[-1-i].quantize()
            bl += 1
        self.qfeature = nn.ModuleList(self.qfeature)

        if not isinstance(self.net.maxpool, nn.Identity):
            self.qmaxpool = QMaxPooling2d(a_bits=self.a_bits, kernel_size=self.net.maxpool.kernel_size, 
                                          stride=self.net.maxpool.stride, padding=self.net.maxpool.padding, 
                                          qi=False, qo=True, signed=self.signed)

        self.qlayer0 = QConvBNReLU(self.net.layer0.conv, self.net.layer0.bn, qi=True, qo=True, w_bits=self.w_bits, a_bits=self.a_bits, signed=self.signed, bl=bl)


    def freeze(self):
        self.qlayer0.freeze( )
        if not isinstance(self.net.maxpool, nn.Identity):
            self.qmaxpool.freeze(qi=self.qlayer0.qo)
            maxpool_qo = self.qmaxpool.qo
        else:
            maxpool_qo = self.qlayer0.qo
        for i, qlayer in enumerate(self.qfeature):
            if i==0:
                qlayer.freeze(pre_qo=maxpool_qo)
            else:
                qlayer.freeze(pre_qo=self.qfeature[i-1].get_last_layer_qo())

        flg = torch.where(torch.tensor(self.net.flag))[0]
  
        self.qneck5.freeze(self.qfeature[flg[3]].get_last_layer_qo())
        self.qneck4.freeze(self.qfeature[flg[2]].get_last_layer_qo())
        self.qneck3.freeze(self.qfeature[flg[1]].get_last_layer_qo())

        self.qadd4.freeze(self.qneck4.get_last_layer_qo(), self.qneck5.get_last_layer_qo())
        self.qadd3.freeze(self.qneck3.get_last_layer_qo(), self.qadd4.qo)



    def quantize_forward(self, x):
        x = self.qlayer0(x)
        x = self.qmaxpool(x)
        out = []
        for qlayer, f in zip(self.qfeature, self.net.flag):
            x = qlayer.quantize_forward(x)
            if f:
                out.append(x)

        if hasattr(self, 'qneck3'):
            p3 = self.qneck3.quantize_forward(out[1])
            p4 = self.qneck4.quantize_forward(out[2])
            p5 = self.qneck5.quantize_forward(out[3])

            p4 = self.qadd4(p4, self.up4(p5))
            p3 = self.qadd3(p3, self.up3(p4))


    def update_quantize_parameters(self):
        self.qlayer0.update_quantize_parameters()
        for i, qlayer in enumerate(self.qfeature):
            qlayer.update_quantize_parameters()

        if hasattr(self, 'qneck3'):
            self.qneck3.update_quantize_parameters()
            self.qneck4.update_quantize_parameters()
            self.qneck5.update_quantize_parameters()

    def quantize_inference(self, x):
        qx = self.qlayer0.qi.quantize_tensor(x)
        qx = self.qlayer0.quantize_inference(qx)
        qx = self.qmaxpool.quantize_inference(qx)
         
        features = []
        for qlayer, f in zip(self.qfeature, self.net.flag):
            qx = qlayer.quantize_inference(qx)
            if f:
                features.append(qx)
    
        p3 = self.qneck3.quantize_inference(features[1])
        p4 = self.qneck4.quantize_inference(features[2])
        p5 = self.qneck5.quantize_inference(features[3])

        p4 = self.qadd4.quantize_inference(p4, self.up4(p5))
        p3 = self.qadd3.quantize_inference(p3, self.up3(p4))

        p3 = self.qadd3.qo.dequantize_tensor(p3)
        p4 = self.qadd4.qo.dequantize_tensor(p4)
        p5 = self.qneck5.get_last_layer_qo().dequantize_tensor(p5)  
        
        cls3, reg3 = self.head3(p3)
        cls4, reg4 = self.head4(p4)
        cls5, reg5 = self.head5(p5)

        cls = [cls3, cls4, cls5]
        reg = [reg3, reg4, reg5]
        return cls, reg

    
    def quantize_training(self, x, record_act):
        q = self.qlayer0.qi.quantize_tensor(x)

        qx = self.qlayer0.quantize_training(q)
        record_act.append(qx - self.qlayer0.qo.zero_point)

        qx = self.qmaxpool.quantize_training(qx)

        features = []
        for qlayer, f in zip(self.qfeature, self.net.flag):  
            qx = qlayer.quantize_training(qx, record_act)
            if f:
                features.append(qx)

        p3 = self.qneck3.quantize_training(features[1], record_act)
        p4 = self.qneck4.quantize_training(features[2], record_act)
        p5 = self.qneck5.quantize_training(features[3], record_act)

        p4 = self.qadd4.quantize_training(p4, self.up4(p5))
        p3 = self.qadd3.quantize_training(p3, self.up3(p4))

        p3 = self.qadd3.qo.dequantize_tensor(p3)
        p4 = self.qadd4.qo.dequantize_tensor(p4)
        p5 = self.qneck5.get_last_layer_qo().dequantize_tensor(p5)

        cls3, reg3 = self.head3(p3)
        cls4, reg4 = self.head4(p4)
        cls5, reg5 = self.head5(p5)

        cls = [cls3, cls4, cls5]
        reg = [reg3, reg4, reg5]
        return cls, reg
 