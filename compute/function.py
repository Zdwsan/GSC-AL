
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch.nn.functional as F
import cv2
import numpy as np
from matplotlib.patches import Rectangle
from torchvision.ops import nms, box_iou
import torch.nn as nn
import math
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import contextlib
import io
import logging
import os
import sys

def make_anchors(feats, strides, grid_cell_offset=0.5):
    """Generate anchors from features."""
    anchor_points, stride_tensor = [], []
    assert feats is not None
    dtype, device = feats[0].dtype, feats[0].device
    for i, stride in enumerate(strides):
        h, w = feats[i].shape[2:] if isinstance(feats, list) else (int(feats[i][0]), int(feats[i][1]))
        sx = torch.arange(end=w, device=device, dtype=dtype) + grid_cell_offset  # shift x
        sy = torch.arange(end=h, device=device, dtype=dtype) + grid_cell_offset  # shift y
        sy, sx = torch.meshgrid(sy, sx, indexing="ij") # if TORCH_1_10 else torch.meshgrid(sy, sx)
        anchor_points.append(torch.stack((sx, sy), -1).view(-1, 2))
        stride_tensor.append(torch.full((h * w, 1), stride, dtype=dtype, device=device))
    return torch.cat(anchor_points), torch.cat(stride_tensor)


def encoder2boxes(reg_pred, stride, imagesize):
    '''
        reg_pred: [B, H, W, 4, grid_size]
    '''
    W, H = imagesize
    B, c, f_h, f_w = reg_pred.shape
    grid_size = c // 4

    device = reg_pred.device

    if grid_size > 1:
        reg_pred = reg_pred.permute(0, 2, 3, 1).view(B, -1, 4, grid_size)
        pred_prob = F.softmax(reg_pred, dim=-1)
        grid = torch.arange(grid_size, device=device)
        # print()
        pred_box = (pred_prob * grid).sum(dim=-1) 
    else:
        pred_box = reg_pred.permute(0, 2, 3, 1).view(B, -1, 4)

    x = torch.arange(f_w, device=device).to(device).float()  # [1,1,f_w]
    y = torch.arange(f_h, device=device).to(device).float()  # [1,f_h,1]
    grid_y, grid_x = torch.meshgrid([y, x], indexing='ij')
    grid_xy = torch.stack([grid_x, grid_y], dim=-1).view(1, -1, 2).to(device)

    x1 = (grid_xy[..., 0] - pred_box[..., 0]) * stride
    y1 = (grid_xy[..., 1] - pred_box[..., 1]) * stride
    x2 = (grid_xy[..., 0] + pred_box[..., 2]) * stride
    y2 = (grid_xy[..., 1] + pred_box[..., 3]) * stride

    boxes = torch.stack([x1, y1, x2, y2], dim=-1).clamp_(0, imagesize[0])
    return boxes

def gt2xyxy(delta_boxes, imagesize):
    device = delta_boxes.device
    W, H = imagesize
    xcenter = delta_boxes[..., 0] * W
    ycenter = delta_boxes[..., 1] * H
    width = delta_boxes[..., 2] * W
    height = delta_boxes[..., 3] * H

    xmin = xcenter - width / 2
    ymin = ycenter - height / 2
    xmax = xcenter + width / 2
    ymax = ycenter + height / 2

    boxes = torch.stack([xmin, ymin, xmax, ymax], dim=-1).clamp(min=0, max=W).to(device)
    return boxes



def visualize_detections(image, conf_threshold, test_target, boxes_list, scores_list, labels_list, class_names, epoch, batch_id, method, dataset):
    Height, Width = image.shape[:2]
    image = np.ascontiguousarray(image)
    fig, ax = plt.subplots(1)
    image = image[..., ::-1]
    ax.imshow(image)
    for n, (boxes, scores, labels) in enumerate(zip(boxes_list, scores_list, labels_list)):
        for i in range(len(boxes)):
            score = scores[i]
            if score < conf_threshold:  
                continue

            box = boxes[i]

            x_min, y_min, x_max, y_max = int(box[0]), int(box[1]), int(box[2]), int(box[3])

            x_min = max(0, x_min) 
            y_min = max(0, y_min) 
            x_max = min(Width, x_max)
            y_max = min(Height, y_max) 

            label = int(labels[i])

            class_name = class_names[label]

            w = x_max - x_min
            h = y_max - y_min
            
            ax.add_patch(plt.Rectangle((x_min, y_min), w, h, color='blue', 
                                       linewidth=1.5, fill=False, alpha=1))
            ax.text(x_min, y_min, 
                    f"{class_name}: {score:.2f}",
                    color='blue', 
                    fontsize=10,
                    alpha=1
            )
        
    target_label = test_target['labels'].cpu().detach().numpy()
    target_boxes = test_target['boxes']
    if target_boxes.shape[0] > 0:
        target_boxes = gt2xyxy(target_boxes, imagesize=(Width, Height)).cpu().detach().numpy()
    
    for la, bo in zip(target_label, target_boxes):
        class_name = class_names[la]

        xmin, ymin, xmax, ymax = map(int, bo)

        w = xmax - xmin
        h = ymax - ymin
        ax.add_patch(plt.Rectangle((xmin, ymin), w, h, color='red', 
                                   linewidth=1.5, fill=False, alpha=0.75))
        ax.text(xmin, ymin, 
                class_name,
                color='red', 
                fontsize=10,
        )
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    os.makedirs("pred/" + method + '-' + dataset, exist_ok=True)
    plt.savefig("pred/" + method + '-' + dataset + "/output-" + str(epoch) + '-' + str(batch_id) + ".jpg")
    plt.close(fig)



def bbox2dist(anchor_points, bbox, reg_max):
    """Transform bbox(xyxy) to dist(ltrb)."""
    x1y1, x2y2 = bbox.chunk(2, -1)
    return torch.cat((anchor_points - x1y1, x2y2 - anchor_points), -1).clamp_(0, reg_max - 0.01)  # dist (lt, rb)


def dist2bbox(distance, anchor_points, xywh=True, dim=-1):
    """Transform distance(ltrb) to box(xywh or xyxy)."""
    lt, rb = distance.chunk(2, dim)
    x1y1 = anchor_points - lt
    x2y2 = anchor_points + rb
    if xywh:
        c_xy = (x1y1 + x2y2) / 2
        wh = x2y2 - x1y1
        return torch.cat((c_xy, wh), dim)  # xywh bbox
    return torch.cat((x1y1, x2y2), dim)  # xyxy bbox


def bbox_decode(anchor_points, pred_dist, reg_max):
    """Decode predicted object bounding box coordinates from anchor points and distribution."""
    if reg_max > 1:
        proj = torch.arange(reg_max, dtype=torch.float, device=pred_dist.device)
        b, a, c = pred_dist.shape  # batch, anchors, channels
        pred_dist = pred_dist.view(b, a, 4, c // 4).softmax(3).matmul(proj.type(pred_dist.dtype))
    return dist2bbox(pred_dist, anchor_points, xywh=False)


def bbox_iou(box1, box2, xywh=True, GIoU=False, DIoU=False, CIoU=False, eps=1e-7):
    if xywh:  # transform from xywh to xyxy
        (x1, y1, w1, h1), (x2, y2, w2, h2) = box1.chunk(4, -1), box2.chunk(4, -1)
        w1_, h1_, w2_, h2_ = w1 / 2, h1 / 2, w2 / 2, h2 / 2
        b1_x1, b1_x2, b1_y1, b1_y2 = x1 - w1_, x1 + w1_, y1 - h1_, y1 + h1_
        b2_x1, b2_x2, b2_y1, b2_y2 = x2 - w2_, x2 + w2_, y2 - h2_, y2 + h2_
    else:  # x1, y1, x2, y2 = box1
        b1_x1, b1_y1, b1_x2, b1_y2 = box1.chunk(4, -1)
        b2_x1, b2_y1, b2_x2, b2_y2 = box2.chunk(4, -1)
        w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps
        w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps

    # Intersection area
    inter = (b1_x2.minimum(b2_x2) - b1_x1.maximum(b2_x1)).clamp_(0) * (
        b1_y2.minimum(b2_y2) - b1_y1.maximum(b2_y1)
    ).clamp_(0)

    # Union Area
    union = w1 * h1 + w2 * h2 - inter + eps

    # IoU
    iou = inter / union
    if CIoU or DIoU or GIoU:
        cw = b1_x2.maximum(b2_x2) - b1_x1.minimum(b2_x1)  # convex (smallest enclosing box) width
        ch = b1_y2.maximum(b2_y2) - b1_y1.minimum(b2_y1)  # convex height
        if CIoU or DIoU:  
            c2 = cw.pow(2) + ch.pow(2) + eps  # convex diagonal squared
            rho2 = (
                (b2_x1 + b2_x2 - b1_x1 - b1_x2).pow(2) + (b2_y1 + b2_y2 - b1_y1 - b1_y2).pow(2)
            ) / 4  
            if CIoU:  
                v = (4 / math.pi**2) * ((w2 / h2).atan() - (w1 / h1).atan()).pow(2)
                with torch.no_grad():
                    alpha = v / (v - iou + (1 + eps))
                return iou - (rho2 / c2 + v * alpha)  # CIoU
            return iou - rho2 / c2  
        c_area = cw * ch + eps 
        return iou - (c_area - union) / c_area 
    return iou  
