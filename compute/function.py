import math
import os

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F


def make_anchors(feats, strides, grid_cell_offset=0.5):
    """Generate anchors from feature maps."""
    anchor_points, stride_tensor = [], []
    assert feats is not None
    dtype, device = feats[0].dtype, feats[0].device
    for i, stride in enumerate(strides):
        h, w = (
            feats[i].shape[2:]
            if isinstance(feats, list)
            else (int(feats[i][0]), int(feats[i][1]))
        )
        sx = torch.arange(end=w, device=device, dtype=dtype) + grid_cell_offset
        sy = torch.arange(end=h, device=device, dtype=dtype) + grid_cell_offset
        sy, sx = torch.meshgrid(sy, sx, indexing="ij")
        anchor_points.append(torch.stack((sx, sy), -1).view(-1, 2))
        stride_tensor.append(torch.full((h * w, 1), stride, dtype=dtype, device=device))
    return torch.cat(anchor_points), torch.cat(stride_tensor)


def encoder2boxes(reg_pred, stride, imagesize):
    """Decode regression output into xyxy boxes."""
    batch_size, channels, feature_h, feature_w = reg_pred.shape
    grid_size = channels // 4
    device = reg_pred.device

    if grid_size > 1:
        reg_pred = reg_pred.permute(0, 2, 3, 1).view(batch_size, -1, 4, grid_size)
        pred_prob = F.softmax(reg_pred, dim=-1)
        grid = torch.arange(grid_size, device=device)
        pred_box = (pred_prob * grid).sum(dim=-1)
    else:
        pred_box = reg_pred.permute(0, 2, 3, 1).view(batch_size, -1, 4)

    x = torch.arange(feature_w, device=device).float()
    y = torch.arange(feature_h, device=device).float()
    grid_y, grid_x = torch.meshgrid(y, x, indexing="ij")
    grid_xy = torch.stack([grid_x, grid_y], dim=-1).view(1, -1, 2).to(device)

    x1 = (grid_xy[..., 0] - pred_box[..., 0]) * stride
    y1 = (grid_xy[..., 1] - pred_box[..., 1]) * stride
    x2 = (grid_xy[..., 0] + pred_box[..., 2]) * stride
    y2 = (grid_xy[..., 1] + pred_box[..., 3]) * stride

    return torch.stack([x1, y1, x2, y2], dim=-1).clamp_(0, imagesize[0])


def gt2xyxy(delta_boxes, imagesize):
    device = delta_boxes.device
    width, height = imagesize
    xcenter = delta_boxes[..., 0] * width
    ycenter = delta_boxes[..., 1] * height
    box_width = delta_boxes[..., 2] * width
    box_height = delta_boxes[..., 3] * height

    xmin = xcenter - box_width / 2
    ymin = ycenter - box_height / 2
    xmax = xcenter + box_width / 2
    ymax = ycenter + box_height / 2

    return (
        torch.stack([xmin, ymin, xmax, ymax], dim=-1).clamp(min=0, max=width).to(device)
    )


def visualize_detections(
    image,
    conf_threshold,
    test_target,
    boxes_list,
    scores_list,
    labels_list,
    class_names,
    epoch,
    batch_id,
    method,
    dataset,
):
    height, width = image.shape[:2]
    image = np.ascontiguousarray(image)
    fig, ax = plt.subplots(1)
    ax.imshow(image[..., ::-1])

    for boxes, scores, labels in zip(boxes_list, scores_list, labels_list):
        for box, score, label in zip(boxes, scores, labels):
            if score < conf_threshold:
                continue

            x_min, y_min, x_max, y_max = map(int, box)
            x_min = max(0, x_min)
            y_min = max(0, y_min)
            x_max = min(width, x_max)
            y_max = min(height, y_max)

            class_name = class_names[int(label)]
            ax.add_patch(
                plt.Rectangle(
                    (x_min, y_min),
                    x_max - x_min,
                    y_max - y_min,
                    color="blue",
                    linewidth=1.5,
                    fill=False,
                    alpha=1,
                )
            )
            ax.text(
                x_min,
                y_min,
                f"{class_name}: {score:.2f}",
                color="blue",
                fontsize=10,
                alpha=1,
            )

    target_label = test_target["labels"].cpu().detach().numpy()
    target_boxes = test_target["boxes"]
    if target_boxes.shape[0] > 0:
        target_boxes = (
            gt2xyxy(target_boxes, imagesize=(width, height)).cpu().detach().numpy()
        )

    for label, box in zip(target_label, target_boxes):
        x_min, y_min, x_max, y_max = map(int, box)
        ax.add_patch(
            plt.Rectangle(
                (x_min, y_min),
                x_max - x_min,
                y_max - y_min,
                color="red",
                linewidth=1.5,
                fill=False,
                alpha=0.75,
            )
        )
        ax.text(x_min, y_min, class_names[label], color="red", fontsize=10)

    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    output_dir = os.path.join("pred", f"{method}-{dataset}")
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, f"output-{epoch}-{batch_id}.jpg"))
    plt.close(fig)


def bbox2dist(anchor_points, bbox, reg_max):
    """Transform xyxy box to left/top/right/bottom distances."""
    x1y1, x2y2 = bbox.chunk(2, -1)
    return torch.cat((anchor_points - x1y1, x2y2 - anchor_points), -1).clamp_(
        0, reg_max - 0.01
    )


def dist2bbox(distance, anchor_points, xywh=True, dim=-1):
    """Transform left/top/right/bottom distances to xywh or xyxy boxes."""
    lt, rb = distance.chunk(2, dim)
    x1y1 = anchor_points - lt
    x2y2 = anchor_points + rb
    if xywh:
        center_xy = (x1y1 + x2y2) / 2
        wh = x2y2 - x1y1
        return torch.cat((center_xy, wh), dim)
    return torch.cat((x1y1, x2y2), dim)


def bbox_decode(anchor_points, pred_dist, reg_max):
    """Decode predicted distribution to box coordinates."""
    if reg_max > 1:
        proj = torch.arange(reg_max, dtype=torch.float, device=pred_dist.device)
        batch_size, anchors, channels = pred_dist.shape
        pred_dist = (
            pred_dist.view(batch_size, anchors, 4, channels // 4)
            .softmax(3)
            .matmul(proj.type(pred_dist.dtype))
        )
    return dist2bbox(pred_dist, anchor_points, xywh=False)


def bbox_iou(box1, box2, xywh=True, GIoU=False, DIoU=False, CIoU=False, eps=1e-7):
    if xywh:
        (x1, y1, w1, h1), (x2, y2, w2, h2) = box1.chunk(4, -1), box2.chunk(4, -1)
        w1_half, h1_half, w2_half, h2_half = w1 / 2, h1 / 2, w2 / 2, h2 / 2
        b1_x1, b1_x2 = x1 - w1_half, x1 + w1_half
        b1_y1, b1_y2 = y1 - h1_half, y1 + h1_half
        b2_x1, b2_x2 = x2 - w2_half, x2 + w2_half
        b2_y1, b2_y2 = y2 - h2_half, y2 + h2_half
    else:
        b1_x1, b1_y1, b1_x2, b1_y2 = box1.chunk(4, -1)
        b2_x1, b2_y1, b2_x2, b2_y2 = box2.chunk(4, -1)
        w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps
        w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps

    inter = (b1_x2.minimum(b2_x2) - b1_x1.maximum(b2_x1)).clamp_(0) * (
        b1_y2.minimum(b2_y2) - b1_y1.maximum(b2_y1)
    ).clamp_(0)
    union = w1 * h1 + w2 * h2 - inter + eps
    iou = inter / union

    if CIoU or DIoU or GIoU:
        convex_w = b1_x2.maximum(b2_x2) - b1_x1.minimum(b2_x1)
        convex_h = b1_y2.maximum(b2_y2) - b1_y1.minimum(b2_y1)
        if CIoU or DIoU:
            convex_diag = convex_w.pow(2) + convex_h.pow(2) + eps
            center_dist = (
                (b2_x1 + b2_x2 - b1_x1 - b1_x2).pow(2)
                + (b2_y1 + b2_y2 - b1_y1 - b1_y2).pow(2)
            ) / 4
            if CIoU:
                v = (4 / math.pi**2) * ((w2 / h2).atan() - (w1 / h1).atan()).pow(2)
                with torch.no_grad():
                    alpha = v / (v - iou + (1 + eps))
                return iou - (center_dist / convex_diag + v * alpha)
            return iou - center_dist / convex_diag
        convex_area = convex_w * convex_h + eps
        return iou - (convex_area - union) / convex_area
    return iou
