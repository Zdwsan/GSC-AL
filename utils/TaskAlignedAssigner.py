
import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from compute.function import *
from torchvision.ops import nms, box_iou

class TaskAlignedAssigner(nn.Module):
    """
    A task-aligned assigner for object detection.

    This class assigns ground-truth (gt) objects to anchors based on the task-aligned metric, which combines both
    classification and localization information.

    Attributes:
        topk (int): The number of top candidates to consider.
        num_classes (int): The number of object classes.
        bg_idx (int): Background class index.
        alpha (float): The alpha parameter for the classification component of the task-aligned metric.
        beta (float): The beta parameter for the localization component of the task-aligned metric.
        eps (float): A small value to prevent division by zero.
    """

    def __init__(self, topk=13, num_classes=80, alpha=1.0, beta=6.0, eps=1e-9):
        """Initialize a TaskAlignedAssigner object with customizable hyperparameters."""
        super().__init__()
        self.topk = topk
        self.num_classes = num_classes
        self.bg_idx = num_classes
        self.alpha = alpha
        self.beta = beta
        self.eps = eps

    @torch.no_grad()
    def forward(self, pd_scores, pd_bboxes, anc_points, gt_labels, gt_bboxes, mask_gt):
        """
        Compute the task-aligned assignment.

        Args:
            pd_scores (torch.Tensor): Predicted classification scores with shape (bs, num_total_anchors, num_classes).
            pd_bboxes (torch.Tensor): Predicted bounding boxes with shape (bs, num_total_anchors, 4).
            anc_points (torch.Tensor): Anchor points with shape (num_total_anchors, 2).
            gt_labels (torch.Tensor): Ground truth labels with shape (bs, n_max_boxes, 1).
            gt_bboxes (torch.Tensor): Ground truth boxes with shape (bs, n_max_boxes, 4).
            mask_gt (torch.Tensor): Mask for valid ground truth boxes with shape (bs, n_max_boxes, 1).

        Returns:
            target_labels (torch.Tensor): Target labels with shape (bs, num_total_anchors).
            target_bboxes (torch.Tensor): Target bounding boxes with shape (bs, num_total_anchors, 4).
            target_scores (torch.Tensor): Target scores with shape (bs, num_total_anchors, num_classes).
            fg_mask (torch.Tensor): Foreground mask with shape (bs, num_total_anchors).
            target_gt_idx (torch.Tensor): Target ground truth indices with shape (bs, num_total_anchors).

        References:
            https://github.com/Nioolek/PPYOLOE_pytorch/blob/master/ppyoloe/assigner/tal_assigner.py
        """
        self.bs = pd_scores.shape[0]
        self.n_max_boxes = gt_bboxes.shape[1]
        device = gt_bboxes.device

        if self.n_max_boxes == 0:
            return (
                torch.full_like(pd_scores[..., 0], self.bg_idx),
                torch.zeros_like(pd_bboxes),
                torch.zeros_like(pd_scores),
                torch.zeros_like(pd_scores[..., 0]),
                torch.zeros_like(pd_scores[..., 0]),
            )

        try:
            return self._forward(pd_scores, pd_bboxes, anc_points, gt_labels, gt_bboxes, mask_gt)
        except torch.OutOfMemoryError:
            # Move tensors to CPU, compute, then move back to original device
            cpu_tensors = [t.cpu() for t in (pd_scores, pd_bboxes, anc_points, gt_labels, gt_bboxes, mask_gt)]
            result = self._forward(*cpu_tensors)
            return tuple(t.to(device) for t in result)

    def _forward(self, pd_scores, pd_bboxes, anc_points, gt_labels, gt_bboxes, mask_gt):
        """
        Compute the task-aligned assignment.

        Args:
            pd_scores (torch.Tensor): Predicted classification scores with shape (bs, num_total_anchors, num_classes).
            pd_bboxes (torch.Tensor): Predicted bounding boxes with shape (bs, num_total_anchors, 4).
            anc_points (torch.Tensor): Anchor points with shape (num_total_anchors, 2).
            gt_labels (torch.Tensor): Ground truth labels with shape (bs, n_max_boxes, 1).
            gt_bboxes (torch.Tensor): Ground truth boxes with shape (bs, n_max_boxes, 4).
            mask_gt (torch.Tensor): Mask for valid ground truth boxes with shape (bs, n_max_boxes, 1).

        Returns:
            target_labels (torch.Tensor): Target labels with shape (bs, num_total_anchors).
            target_bboxes (torch.Tensor): Target bounding boxes with shape (bs, num_total_anchors, 4).
            target_scores (torch.Tensor): Target scores with shape (bs, num_total_anchors, num_classes).
            fg_mask (torch.Tensor): Foreground mask with shape (bs, num_total_anchors).
            target_gt_idx (torch.Tensor): Target ground truth indices with shape (bs, num_total_anchors).
        """
        mask_pos, align_metric, overlaps = self.get_pos_mask(
            pd_scores, pd_bboxes, gt_labels, gt_bboxes, anc_points, mask_gt
        )

        target_gt_idx, fg_mask, mask_pos = self.select_highest_overlaps(mask_pos, overlaps, self.n_max_boxes)

        # Assigned target
        target_labels, target_bboxes, target_scores = self.get_targets(gt_labels, gt_bboxes, target_gt_idx, fg_mask)

        # Normalize
        align_metric *= mask_pos
        pos_align_metrics = align_metric.amax(dim=-1, keepdim=True)  # b, max_num_obj
        pos_overlaps = (overlaps * mask_pos).amax(dim=-1, keepdim=True)  # b, max_num_obj
        norm_align_metric = (align_metric * pos_overlaps / (pos_align_metrics + self.eps)).amax(-2).unsqueeze(-1)
        target_scores = target_scores * norm_align_metric

        return target_labels, target_bboxes, target_scores, fg_mask.bool(), target_gt_idx

    def get_pos_mask(self, pd_scores, pd_bboxes, gt_labels, gt_bboxes, anc_points, mask_gt):
        """
        Get positive mask for each ground truth box.

        Args:
            pd_scores (torch.Tensor): Predicted classification scores with shape (bs, num_total_anchors, num_classes).
            pd_bboxes (torch.Tensor): Predicted bounding boxes with shape (bs, num_total_anchors, 4).
            gt_labels (torch.Tensor): Ground truth labels with shape (bs, n_max_boxes, 1).
            gt_bboxes (torch.Tensor): Ground truth boxes with shape (bs, n_max_boxes, 4).
            anc_points (torch.Tensor): Anchor points with shape (num_total_anchors, 2).
            mask_gt (torch.Tensor): Mask for valid ground truth boxes with shape (bs, n_max_boxes, 1).

        Returns:
            mask_pos (torch.Tensor): Positive mask with shape (bs, max_num_obj, h*w).
            align_metric (torch.Tensor): Alignment metric with shape (bs, max_num_obj, h*w).
            overlaps (torch.Tensor): Overlaps between predicted and ground truth boxes with shape (bs, max_num_obj, h*w).
        """
        mask_in_gts = self.select_candidates_in_gts(anc_points, gt_bboxes)
        # Get anchor_align metric, (b, max_num_obj, h*w)
        align_metric, overlaps = self.get_box_metrics(pd_scores, pd_bboxes, gt_labels, gt_bboxes, mask_in_gts * mask_gt)
        # Get topk_metric mask, (b, max_num_obj, h*w)
        mask_topk = self.select_topk_candidates(align_metric, topk_mask=mask_gt.expand(-1, -1, self.topk).bool())
        # Merge all mask to a final mask, (b, max_num_obj, h*w)
        mask_pos = mask_topk * mask_in_gts * mask_gt

        return mask_pos, align_metric, overlaps

    def get_box_metrics(self, pd_scores, pd_bboxes, gt_labels, gt_bboxes, mask_gt):
        """
        Compute alignment metric given predicted and ground truth bounding boxes.

        Args:
            pd_scores (torch.Tensor): Predicted classification scores with shape (bs, num_total_anchors, num_classes).
            pd_bboxes (torch.Tensor): Predicted bounding boxes with shape (bs, num_total_anchors, 4).
            gt_labels (torch.Tensor): Ground truth labels with shape (bs, n_max_boxes, 1).
            gt_bboxes (torch.Tensor): Ground truth boxes with shape (bs, n_max_boxes, 4).
            mask_gt (torch.Tensor): Mask for valid ground truth boxes with shape (bs, n_max_boxes, h*w).

        Returns:
            align_metric (torch.Tensor): Alignment metric combining classification and localization.
            overlaps (torch.Tensor): IoU overlaps between predicted and ground truth boxes.
        """
        na = pd_bboxes.shape[-2]
        mask_gt = mask_gt.bool()  # b, max_num_obj, h*w
        overlaps = torch.zeros([self.bs, self.n_max_boxes, na], dtype=pd_bboxes.dtype, device=pd_bboxes.device)
        bbox_scores = torch.zeros([self.bs, self.n_max_boxes, na], dtype=pd_scores.dtype, device=pd_scores.device)

        ind = torch.zeros([2, self.bs, self.n_max_boxes], dtype=torch.long)  # 2, b, max_num_obj
        ind[0] = torch.arange(end=self.bs).view(-1, 1).expand(-1, self.n_max_boxes)  # b, max_num_obj
        ind[1] = gt_labels.squeeze(-1)  # b, max_num_obj
        # Get the scores of each grid for each gt cls
        # print(bbox_scores.shape, mask_gt.shape, pd_scores.shape, ind[0].shape, ind[1].shape)
        bbox_scores[mask_gt] = pd_scores[ind[0], :, ind[1]][mask_gt]  # b, max_num_obj, h*w

        # (b, max_num_obj, 1, 4), (b, 1, h*w, 4)
        pd_boxes = pd_bboxes.unsqueeze(1).expand(-1, self.n_max_boxes, -1, -1)[mask_gt]
        gt_boxes = gt_bboxes.unsqueeze(2).expand(-1, -1, na, -1)[mask_gt]
        overlaps[mask_gt] = self.iou_calculation(gt_boxes, pd_boxes)

        align_metric = bbox_scores.pow(self.alpha) * overlaps.pow(self.beta)
        return align_metric, overlaps

    def iou_calculation(self, gt_bboxes, pd_bboxes):
        """
        Calculate IoU for horizontal bounding boxes.

        Args:
            gt_bboxes (torch.Tensor): Ground truth boxes.
            pd_bboxes (torch.Tensor): Predicted boxes.

        Returns:
            (torch.Tensor): IoU values between each pair of boxes.
        """
        return bbox_iou(gt_bboxes, pd_bboxes, xywh=False, CIoU=True).squeeze(-1).clamp_(0)

    def select_topk_candidates(self, metrics, largest=True, topk_mask=None):
        """
        Select the top-k candidates based on the given metrics.

        Args:
            metrics (torch.Tensor): A tensor of shape (b, max_num_obj, h*w), where b is the batch size,
                              max_num_obj is the maximum number of objects, and h*w represents the
                              total number of anchor points.
            largest (bool): If True, select the largest values; otherwise, select the smallest values.
            topk_mask (torch.Tensor): An optional boolean tensor of shape (b, max_num_obj, topk), where
                                topk is the number of top candidates to consider. If not provided,
                                the top-k values are automatically computed based on the given metrics.

        Returns:
            (torch.Tensor): A tensor of shape (b, max_num_obj, h*w) containing the selected top-k candidates.
        """
        # (b, max_num_obj, topk)
        topk_metrics, topk_idxs = torch.topk(metrics, self.topk, dim=-1, largest=largest)
        if topk_mask is None:
            topk_mask = (topk_metrics.max(-1, keepdim=True)[0] > self.eps).expand_as(topk_idxs)
        # (b, max_num_obj, topk)
        topk_idxs.masked_fill_(~topk_mask, 0)

        # (b, max_num_obj, topk, h*w) -> (b, max_num_obj, h*w)
        count_tensor = torch.zeros(metrics.shape, dtype=torch.int8, device=topk_idxs.device)
        ones = torch.ones_like(topk_idxs[:, :, :1], dtype=torch.int8, device=topk_idxs.device)
        for k in range(self.topk):
            # Expand topk_idxs for each value of k and add 1 at the specified positions
            count_tensor.scatter_add_(-1, topk_idxs[:, :, k : k + 1], ones)
        # Filter invalid bboxes
        count_tensor.masked_fill_(count_tensor > 1, 0)

        return count_tensor.to(metrics.dtype)

    def get_targets(self, gt_labels, gt_bboxes, target_gt_idx, fg_mask):
        """
        Compute target labels, target bounding boxes, and target scores for the positive anchor points.

        Args:
            gt_labels (torch.Tensor): Ground truth labels of shape (b, max_num_obj, 1), where b is the
                                batch size and max_num_obj is the maximum number of objects.
            gt_bboxes (torch.Tensor): Ground truth bounding boxes of shape (b, max_num_obj, 4).
            target_gt_idx (torch.Tensor): Indices of the assigned ground truth objects for positive
                                    anchor points, with shape (b, h*w), where h*w is the total
                                    number of anchor points.
            fg_mask (torch.Tensor): A boolean tensor of shape (b, h*w) indicating the positive
                              (foreground) anchor points.

        Returns:
            target_labels (torch.Tensor): Shape (b, h*w), containing the target labels for positive anchor points.
            target_bboxes (torch.Tensor): Shape (b, h*w, 4), containing the target bounding boxes for positive
                                          anchor points.
            target_scores (torch.Tensor): Shape (b, h*w, num_classes), containing the target scores for positive
                                          anchor points.
        """
        # Assigned target labels, (b, 1)
        batch_ind = torch.arange(end=self.bs, dtype=torch.int64, device=gt_labels.device)[..., None]
        target_gt_idx = target_gt_idx + batch_ind * self.n_max_boxes  # (b, h*w)
        target_labels = gt_labels.long().flatten()[target_gt_idx]  # (b, h*w)

        # Assigned target boxes, (b, max_num_obj, 4) -> (b, h*w, 4)
        target_bboxes = gt_bboxes.view(-1, gt_bboxes.shape[-1])[target_gt_idx]

        # Assigned target scores
        target_labels.clamp_(0)

        # 10x faster than F.one_hot()
        target_scores = torch.zeros(
            (target_labels.shape[0], target_labels.shape[1], self.num_classes),
            dtype=torch.int64,
            device=target_labels.device,
        )  # (b, h*w, 80)
        target_scores.scatter_(2, target_labels.unsqueeze(-1), 1)

        fg_scores_mask = fg_mask[:, :, None].repeat(1, 1, self.num_classes)  # (b, h*w, 80)
        target_scores = torch.where(fg_scores_mask > 0, target_scores, 0)

        return target_labels, target_bboxes, target_scores

    @staticmethod
    def select_candidates_in_gts(xy_centers, gt_bboxes, eps=1e-9):
        """
        Select positive anchor centers within ground truth bounding boxes.

        Args:
            xy_centers (torch.Tensor): Anchor center coordinates, shape (h*w, 2).
            gt_bboxes (torch.Tensor): Ground truth bounding boxes, shape (b, n_boxes, 4).
            eps (float, optional): Small value for numerical stability. Defaults to 1e-9.

        Returns:
            (torch.Tensor): Boolean mask of positive anchors, shape (b, n_boxes, h*w).

        Note:
            b: batch size, n_boxes: number of ground truth boxes, h: height, w: width.
            Bounding box format: [x_min, y_min, x_max, y_max].
        """
        n_anchors = xy_centers.shape[0]
        bs, n_boxes, _ = gt_bboxes.shape
        lt, rb = gt_bboxes.view(-1, 1, 4).chunk(2, 2)  # left-top, right-bottom
        bbox_deltas = torch.cat((xy_centers[None] - lt, rb - xy_centers[None]), dim=2).view(bs, n_boxes, n_anchors, -1)
        return bbox_deltas.amin(3).gt_(eps)

    @staticmethod
    def select_highest_overlaps(mask_pos, overlaps, n_max_boxes):
        """
        Select anchor boxes with highest IoU when assigned to multiple ground truths.

        Args:
            mask_pos (torch.Tensor): Positive mask, shape (b, n_max_boxes, h*w).
            overlaps (torch.Tensor): IoU overlaps, shape (b, n_max_boxes, h*w).
            n_max_boxes (int): Maximum number of ground truth boxes.

        Returns:
            target_gt_idx (torch.Tensor): Indices of assigned ground truths, shape (b, h*w).
            fg_mask (torch.Tensor): Foreground mask, shape (b, h*w).
            mask_pos (torch.Tensor): Updated positive mask, shape (b, n_max_boxes, h*w).
        """
        # Convert (b, n_max_boxes, h*w) -> (b, h*w)
        fg_mask = mask_pos.sum(-2)
        if fg_mask.max() > 1:  # one anchor is assigned to multiple gt_bboxes
            mask_multi_gts = (fg_mask.unsqueeze(1) > 1).expand(-1, n_max_boxes, -1)  # (b, n_max_boxes, h*w)
            max_overlaps_idx = overlaps.argmax(1)  # (b, h*w)

            is_max_overlaps = torch.zeros(mask_pos.shape, dtype=mask_pos.dtype, device=mask_pos.device)
            is_max_overlaps.scatter_(1, max_overlaps_idx.unsqueeze(1), 1)

            mask_pos = torch.where(mask_multi_gts, is_max_overlaps, mask_pos).float()  # (b, n_max_boxes, h*w)
            fg_mask = mask_pos.sum(-2)
        # Find each grid serve which gt(index)
        target_gt_idx = mask_pos.argmax(-2)  # (b, h*w)
        return target_gt_idx, fg_mask, mask_pos
    

