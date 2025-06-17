
import torch.nn as nn
import torch
from torchvision.ops import nms, box_iou
from compute.function import *
from utils.TaskAlignedAssigner import *
from utils.loss import *

class ConvBNSiLU(nn.Module):
    """ 标准卷积块：Conv + BN + SiLU激活 """
    def __init__(self, in_c, out_c, kernel=1, stride=1, padding=None, groups=1):
        super().__init__()
        self.conv = nn.Conv2d(in_c, out_c, kernel, stride, 
                             padding=(kernel // 2) if padding is None else padding,
                             groups=groups,
                             bias=False)
        self.bn = nn.BatchNorm2d(out_c)
        self.act = nn.SiLU()
        # self.act = nn.ReLU()

        # self._initial_()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))
    
    def _initial_(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.normal_(m.weight.data, 0.0, 0.01)
            if isinstance(m, (nn.BatchNorm2d)):
                nn.init.constant_(m.weight, 0.1)
                nn.init.constant_(m.bias, 0)

class YOLOv8Head(nn.Module):
    """ YOLOv8解耦检测头 """
    def __init__(self, in_channels, num_classes=80, reg_max=16):
        super().__init__()

        self.num_classes = num_classes
        self.reg_max = reg_max

        # 分类分支
        self.cls_branch = nn.Sequential(
            # ConvBNSiLU(in_channels, in_channels, 3, groups=in_channels),
            # ConvBNSiLU(in_channels, in_channels, 3, groups=in_channels),
            nn.Conv2d(in_channels, num_classes, 1)
        )
        # self.cls_branch = nn.Conv2d(in_channels, num_classes, kernel_size=1)

        # 回归分支（xywh）
        self.reg_branch = nn.Sequential(
            # ConvBNSiLU(in_channels, in_channels, kernel=3, groups=in_channels),
            # ConvBNSiLU(in_channels, in_channels, kernel=3, groups=in_channels),
            nn.Conv2d(in_channels, 4 * reg_max, kernel_size=1)
            # nn.Conv2d(in_channels, 4*num_classes, kernel_size=1)
        )
        # self.reg_branch = nn.Conv2d(in_channels, 4, kernel_size=1)

        # self._initial_()

    def forward(self, x):
        cls_out = self.cls_branch(x)
        reg_out = self.reg_branch(x)
        # B, _, H, W = reg_out.shape
        reg_out = reg_out
        return cls_out, reg_out
    
    def _initial_(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.normal_(m.weight.data, 0.0, 0.01)
            if isinstance(m, (nn.BatchNorm2d)):
                nn.init.constant_(m.weight, 0.1)
                nn.init.constant_(m.bias, 0)



class block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, groups, activate=True):
        super(block, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, groups=groups, bias=False)  # bias False
        self.bn = nn.BatchNorm2d(out_channels, momentum=0.1)
        self.activate = activate
        if activate:
            self.relu = nn.ReLU6()
        # else:
        #     self.relu = nn.Identity()
    def forward(self, x):
        # print(x.shape, self.conv.weight.shape)
        out = self.conv(x)
        out = self.bn(out)
        if self.activate:
            out = self.relu(out)
        return out

class YOLOv8Headtest(nn.Module):
    """ YOLOv8解耦检测头 """
    def __init__(self, in_channels, num_classes=80, reg_max=16):
        super().__init__()

        self.num_classes = num_classes
        self.reg_max = reg_max

        # 分类分支
        self.cls_layer1 = block(in_channels, in_channels, kernel_size=3, stride=1, padding=1, groups=in_channels, activate=True)
        self.cls_layer2 = block(in_channels, in_channels, kernel_size=3, stride=1, padding=1, groups=in_channels, activate=True)
        self.cls_out = nn.Conv2d(in_channels, num_classes, 1)

        # 回归分支（xywh）
        self.reg_layer1 = block(in_channels, in_channels, kernel_size=3, stride=1, padding=1, groups=in_channels, activate=True)
        self.reg_layer2 = block(in_channels, in_channels, kernel_size=3, stride=1, padding=1, groups=in_channels, activate=True)
        self.reg_out = nn.Conv2d(in_channels, 4 * reg_max, kernel_size=1)


    def forward(self, x):
        cls_out = self.cls_layer1(x)
        cls_out = self.cls_layer2(cls_out)
        cls_out = self.cls_out(cls_out)

        reg_out = self.reg_layer1(x)
        reg_out = self.reg_layer2(reg_out)
        reg_out = self.reg_out(reg_out)
        return cls_out, reg_out


class YOLOv8Loss(nn.Module):
    """ YOLOv8损失函数 """
    def __init__(self, num_classes, imagesize, reg_max=16, topk=10, strides=None, device='cuda'):
        super().__init__()
        self.targetGenerator = TaskAlignedAssigner(topk=topk, num_classes=num_classes, alpha=0.5, beta=6.0)
        self.bce = nn.BCEWithLogitsLoss(reduction="none")
        self.bbox_loss = BboxLoss(reg_max).to(device)
        self.reg_max = reg_max
        self.device = device
        self.strides = strides
        self.imagesize = imagesize

    def forward(self, cls_out, reg_out, targets):
        B, num_classes, _, _ = cls_out[0].shape

        anchor_points, stride_tensor = make_anchors(reg_out, self.strides, 0.0)

        pred_scores = torch.cat([cls.permute(0, 2, 3, 1).reshape(B, -1, num_classes) for cls in cls_out], dim=1)
        pred_distri = torch.cat([reg.permute(0, 2, 3, 1).reshape(B, -1, 4*self.reg_max) for reg in reg_out], dim=1)
        gt_bboxes = targets['boxes'].to(self.device)
        gt_bboxes = gt2xyxy(gt_bboxes, (self.imagesize, self.imagesize))

        # 由于填充了标签，所以要获取有效d标签的mask
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0.0)
        
        gt_labels = targets['labels'].to(self.device)
        # print(anchor_points.shape, pred_distri.shape)
        pred_bboxes = bbox_decode(anchor_points, pred_distri, reg_max=self.reg_max)
        # print(pred_bboxes.shape)
        _, target_bboxes, target_scores, fg_mask, _ = self.targetGenerator(
            pred_scores.detach().sigmoid(),
            (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor,
            gt_labels.unsqueeze(-1),
            gt_bboxes,
            mask_gt,
        )

        target_scores_sum = max(target_scores.sum(), 1)

        # 计算损失
        dtype = pred_scores.dtype
        loss = torch.zeros(size=[3], device=self.device)
        # print(pred_scores, self.bce(pred_scores, target_scores.to(dtype)).shape, target_scores_sum)
        loss[0] = self.bce(pred_scores, target_scores.to(dtype)).sum() / target_scores_sum  # BCE

        # Bbox loss
        if fg_mask.sum():
            target_bboxes /= stride_tensor
            loss[1], loss[2] = self.bbox_loss(
                pred_distri, pred_bboxes, anchor_points, target_bboxes, target_scores, target_scores_sum, fg_mask
            )

        loss[0] *= 0.5
        loss[1] *= 7.5
        loss[2] *= 1.5
        
        loss_ = loss.sum() #* B
        return loss_, loss

        

    









    