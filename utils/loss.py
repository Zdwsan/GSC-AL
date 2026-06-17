import torch
import torch.nn as nn
import torch.nn.functional as F

from compute.function import bbox2dist, bbox_iou


class DFLoss(nn.Module):
    """Distribution Focal Loss."""

    def __init__(self, reg_max=16):
        super().__init__()
        self.reg_max = reg_max

    def __call__(self, pred_dist, target):
        target = target.clamp_(0, self.reg_max - 1 - 0.01)
        target_left = target.long()
        target_right = target_left + 1
        weight_left = target_right - target
        weight_right = 1 - weight_left
        return (
            F.cross_entropy(pred_dist, target_left.view(-1), reduction="none").view(
                target_left.shape
            )
            * weight_left
            + F.cross_entropy(pred_dist, target_right.view(-1), reduction="none").view(
                target_left.shape
            )
            * weight_right
        ).mean(-1, keepdim=True)


class BboxLoss(nn.Module):
    """Bounding box IoU and DFL losses."""

    def __init__(self, reg_max=16):
        super().__init__()
        self.dfl_loss = DFLoss(reg_max) if reg_max > 1 else None

    def forward(
        self,
        pred_dist,
        pred_bboxes,
        anchor_points,
        target_bboxes,
        target_scores,
        target_scores_sum,
        fg_mask,
    ):
        weight = target_scores.sum(-1)[fg_mask].unsqueeze(-1)
        iou = bbox_iou(
            pred_bboxes[fg_mask], target_bboxes[fg_mask], xywh=False, CIoU=True
        )
        loss_iou = ((1.0 - iou) * weight).sum() / target_scores_sum

        if self.dfl_loss:
            target_ltrb = bbox2dist(
                anchor_points, target_bboxes, self.dfl_loss.reg_max - 1
            )
            loss_dfl = (
                self.dfl_loss(
                    pred_dist[fg_mask].view(-1, self.dfl_loss.reg_max),
                    target_ltrb[fg_mask],
                )
                * weight
            )
            loss_dfl = loss_dfl.sum() / target_scores_sum
        else:
            loss_dfl = torch.tensor(0.0, device=pred_dist.device)

        return loss_iou, loss_dfl
