import json
import os
import sys

import numpy as np
import torch
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from torchvision.ops import nms

from compute.function import encoder2boxes, gt2xyxy, visualize_detections
from utils.categories import category_name


def test(
    model,
    dataset,
    method,
    num_class,
    test_loader,
    strides,
    imagesize,
    epoch,
    device,
    conf_thres=0.25,
    plot=False,
):
    model.eval()

    images_ann = []
    categories = [{"id": i, "name": f"class_{i}"} for i in range(num_class)]
    annotations = []
    predictions = []
    img_id = 0
    ann_id = 0

    with torch.no_grad():
        for batch_idx, (images, targets) in enumerate(test_loader):
            test_images, model_images = images
            for test_image, model_image, target in zip(
                test_images, model_images, targets
            ):
                images_ann.append(
                    {"id": img_id, "width": imagesize, "height": imagesize}
                )

                image = model_image.unsqueeze(0).to(device)
                target["boxes"] = target["boxes"].to(device)
                target["labels"] = target["labels"].to(device)

                if len(target["boxes"]) > 0:
                    boxes_gt = gt2xyxy(
                        target["boxes"], imagesize=(imagesize, imagesize)
                    )
                    x = boxes_gt[:, 0]
                    y = boxes_gt[:, 1]
                    w = boxes_gt[:, 2] - boxes_gt[:, 0]
                    h = boxes_gt[:, 3] - boxes_gt[:, 1]

                    for label_idx in range(len(target["labels"])):
                        annotations.append(
                            {
                                "id": ann_id,
                                "image_id": img_id,
                                "category_id": target["labels"][label_idx].item(),
                                "bbox": [
                                    x[label_idx].item(),
                                    y[label_idx].item(),
                                    w[label_idx].item(),
                                    h[label_idx].item(),
                                ],
                                "area": w[label_idx].item() * h[label_idx].item(),
                                "iscrowd": 0,
                            }
                        )
                        ann_id += 1

                if method == "GSC_AL":
                    cls_out, reg_out = model.quantize_inference(image)
                elif method == "QAT":
                    cls_out, reg_out = model.quantize_forward(image)
                else:
                    cls_out, reg_out = model(image)

                boxes, labels, scores = [], [], []
                for cls_pred, reg_pred, stride in zip(cls_out, reg_out, strides):
                    batch_size, channels, height, width = cls_pred.shape
                    cls_pred = cls_pred.permute(0, 2, 3, 1).view(
                        batch_size, -1, channels
                    )

                    scores_pred = torch.sigmoid(cls_pred).max(dim=-1).values
                    labels_pred = torch.argmax(cls_pred, dim=-1)
                    boxes_encode = encoder2boxes(
                        reg_pred, stride, (imagesize, imagesize)
                    )

                    boxes.append(boxes_encode.view(-1, 4))
                    labels.append(labels_pred.view(-1))
                    scores.append(scores_pred.view(-1))

                boxes = torch.cat(boxes)
                labels = torch.cat(labels)
                scores = torch.cat(scores)

                class_names = category_name[dataset]
                boxes_nms, scores_nms, labels_nms = [], [], []
                nms_threshold = 0.1

                for label in range(num_class):
                    mask = (labels == label) & (scores >= conf_thres)
                    if mask.sum() == 0:
                        continue
                    indices = class_nms_indices(
                        boxes[mask],
                        scores[mask],
                        iou_threshold=nms_threshold,
                    )
                    boxes_nms.append(boxes[mask][indices])
                    scores_nms.append(scores[mask][indices])
                    labels_nms.append(labels[mask][indices])

                if boxes_nms:
                    boxes = torch.cat(boxes_nms)
                    scores = torch.cat(scores_nms)
                    labels = torch.cat(labels_nms)
                else:
                    boxes = torch.empty((0, 4), device=device)
                    scores = torch.empty((0,), device=device)
                    labels = torch.empty((0,), dtype=torch.long, device=device)

                box_wh = boxes[:, 2:] - boxes[:, 0:2]
                for box, wh, score, label in zip(boxes, box_wh, scores, labels):
                    predictions.append(
                        {
                            "image_id": img_id,
                            "category_id": label.item(),
                            "bbox": [
                                box[0].item(),
                                box[1].item(),
                                wh[0].item(),
                                wh[1].item(),
                            ],
                            "score": score.item(),
                        }
                    )
                img_id += 1

                boxes_nms_np = [item.cpu().detach().numpy() for item in boxes_nms]
                scores_nms_np = [item.cpu().detach().numpy() for item in scores_nms]
                labels_nms_np = [item.cpu().detach().numpy() for item in labels_nms]

                if batch_idx > 4:
                    continue
                if plot:
                    visualize_detections(
                        test_image.permute(1, 2, 0),
                        0.5,
                        target,
                        boxes_nms_np,
                        scores_nms_np,
                        labels_nms_np,
                        class_names,
                        epoch,
                        batch_idx,
                        method,
                        dataset,
                    )

    if len(predictions) == 0:
        index = {"precision": 0, "recall": 0, "F1": 0, "AP": 0, "mAP5095": 0}
    else:
        coco_format = {
            "images": images_ann,
            "annotations": annotations,
            "categories": categories,
        }
        with open("annotations.json", "w", encoding="utf-8") as file:
            json.dump(convert_to_python_type(coco_format), file, indent=2)
        with open("predictions.json", "w", encoding="utf-8") as file:
            json.dump(convert_to_python_type(predictions), file, indent=2)

        index = evaluate_coco_metrics("annotations.json", "predictions.json")

    print(index)
    return index


def calculate_detection_metrics(coco_eval, iou_thr=0.5):
    coco_eval.params.iouThrs = np.array([iou_thr])
    coco_eval.params.areaRng = [[0, 1e5**2]]
    coco_eval.params.maxDets = [100]
    coco_eval.evaluate()

    true_positive, false_positive, false_negative = 0, 0, 0

    for eval_img in coco_eval.evalImgs:
        if eval_img is None:
            continue
        dt_matches = eval_img["dtMatches"][0]
        dt_ignore = eval_img["dtIgnore"][0]
        gt_ignore = eval_img["gtIgnore"]
        gt_matches = eval_img["gtMatches"][0]

        for idx, dt_match in enumerate(dt_matches):
            if dt_ignore[idx]:
                continue
            if dt_match > 0:
                true_positive += 1
            else:
                false_positive += 1

        for idx, gt_match in enumerate(gt_matches):
            if gt_ignore[idx]:
                continue
            if gt_match == 0:
                false_negative += 1

    precision = (
        true_positive / (true_positive + false_positive)
        if (true_positive + false_positive) > 0
        else 0
    )
    recall = (
        true_positive / (true_positive + false_negative)
        if (true_positive + false_negative) > 0
        else 0
    )
    f1 = (
        2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    )

    return true_positive, false_positive, false_negative, precision, recall, f1


def evaluate_coco_metrics(gt_json_path, pred_json_path):
    with open(os.devnull, "w", encoding="utf-8") as devnull:
        original_stdout = sys.stdout
        sys.stdout = devnull
        try:
            coco_gt = COCO(gt_json_path)
            coco_dt = coco_gt.loadRes(pred_json_path)

            coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
            _, _, _, precision, recall, f1 = calculate_detection_metrics(
                coco_eval,
                iou_thr=0.5,
            )

            coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
            coco_eval.evaluate()
            coco_eval.accumulate()
            coco_eval.summarize()

            stats = coco_eval.stats
            return {
                "precision": round(precision * 100, 2),
                "recall": round(recall * 100, 2),
                "F1": round(f1 * 100, 2),
                "AP": round(stats[1] * 100, 2),
                "mAP5095": round(stats[0] * 100, 2),
            }
        finally:
            sys.stdout = original_stdout


def convert_to_python_type(obj):
    if isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    if isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {key: convert_to_python_type(value) for key, value in obj.items()}
    if isinstance(obj, list):
        return [convert_to_python_type(item) for item in obj]
    return obj


def class_nms_indices(boxes, scores, iou_threshold):
    return nms(boxes, scores, iou_threshold)
