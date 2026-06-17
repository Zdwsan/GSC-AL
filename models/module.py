import torch
import torch.nn as nn

from compute.function import bbox_decode, gt2xyxy, make_anchors
from utils.TaskAlignedAssigner import TaskAlignedAssigner
from utils.loss import BboxLoss


class YOLOv8Loss(nn.Module):
    """YOLOv8 detection loss."""

    def __init__(
        self, num_classes, imagesize, reg_max=16, topk=10, strides=None, device="cuda"
    ):
        super().__init__()
        self.targetGenerator = TaskAlignedAssigner(
            topk=topk,
            num_classes=num_classes,
            alpha=0.5,
            beta=6.0,
        )
        self.bce = nn.BCEWithLogitsLoss(reduction="none")
        self.bbox_loss = BboxLoss(reg_max).to(device)
        self.reg_max = reg_max
        self.device = device
        self.strides = strides
        self.imagesize = imagesize

    def forward(self, cls_out, reg_out, targets):
        batch_size, num_classes, _, _ = cls_out[0].shape

        anchor_points, stride_tensor = make_anchors(reg_out, self.strides, 0.0)

        pred_scores = torch.cat(
            [
                cls.permute(0, 2, 3, 1).reshape(batch_size, -1, num_classes)
                for cls in cls_out
            ],
            dim=1,
        )
        pred_distri = torch.cat(
            [
                reg.permute(0, 2, 3, 1).reshape(batch_size, -1, 4 * self.reg_max)
                for reg in reg_out
            ],
            dim=1,
        )
        gt_bboxes = targets["boxes"].to(self.device)
        gt_bboxes = gt2xyxy(gt_bboxes, (self.imagesize, self.imagesize))
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0.0)

        gt_labels = targets["labels"].to(self.device)
        pred_bboxes = bbox_decode(anchor_points, pred_distri, reg_max=self.reg_max)
        _, target_bboxes, target_scores, fg_mask, _ = self.targetGenerator(
            pred_scores.detach().sigmoid(),
            (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor,
            gt_labels.unsqueeze(-1),
            gt_bboxes,
            mask_gt,
        )

        target_scores_sum = max(target_scores.sum(), 1)
        dtype = pred_scores.dtype
        loss = torch.zeros(size=[3], device=self.device)
        loss[0] = (
            self.bce(pred_scores, target_scores.to(dtype)).sum() / target_scores_sum
        )

        if fg_mask.sum():
            target_bboxes /= stride_tensor
            loss[1], loss[2] = self.bbox_loss(
                pred_distri,
                pred_bboxes,
                anchor_points,
                target_bboxes,
                target_scores,
                target_scores_sum,
                fg_mask,
            )

        loss[0] *= 0.5
        loss[1] *= 7.5
        loss[2] *= 1.5

        return loss.sum(), loss
