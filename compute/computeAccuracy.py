import torch
import json
import os
import sys

from compute.function import *
from utils.categories import *
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from torchvision.ops import nms



def test(model, dataset, method, num_class, test_loader, strides, imagesize, epoch, device, conf_thres=0.25, plot=False):
    # 
    model.eval()
 
    TP, FP, FN = [], [], []
    mets = {}
    
    images_ann = []
    categories = [{"id": i, "name": f"class_{i}"} for i in range(num_class)]
    annotations = []

    predictions = []
    img_id = 0
    ann_id = 0

    with torch.no_grad():
        for batch_idx, (images, targets) in enumerate(test_loader):
            o_imgs, imgs = images
            for test_image, test_trans_image, target in zip(o_imgs, imgs, targets):

                temp_img = {"id": img_id, "width":imagesize, "height":imagesize}
                images_ann.append(temp_img)


                if len(test_image.shape) == 3:
                    image = test_trans_image.unsqueeze(0)

                image = image.to(device)
        
                target['boxes'] = target['boxes'].to(device)
                target['labels'] = target['labels'].to(device)

                if len(target['boxes']) == 0:
                    boxes_gt = torch.tensor([]).to(device)
                else:
                    boxes_gt = gt2xyxy(target['boxes'], imagesize=(imagesize, imagesize))

                    x = boxes_gt[:, 0] 
                    y = boxes_gt[:, 1]
                    w = boxes_gt[:, 2] - boxes_gt[:, 0]
                    h = boxes_gt[:, 3] - boxes_gt[:, 1]

                    for nl in range(len(target['labels'])):
                    
                        temp_ann = {"id": ann_id, 
                                    "image_id": img_id,
                                    "category_id": target['labels'][nl].item(),
                                    "bbox": [x[nl].item(), y[nl].item(), w[nl].item(), h[nl].item()],
                                    "area": w[nl].item() * h[nl].item(),
                                    "iscrowd": 0
                                    }
                        annotations.append(temp_ann)
                        ann_id += 1

                if hasattr(model, 'quantize_inference'):
                    cls_out, reg_out = model.quantize_inference(image)
                else:
                    cls_out, reg_out = model(image)

                boxes = []
                labels = []
                scores = []

                count = 0
                for cls_pred, reg_pred, stride in zip(cls_out, reg_out, strides):
                    B, C, H, W = cls_pred.shape
                    cls_pred = cls_pred.permute(0, 2, 3, 1).view(B, -1, C)
 
                    scores_pred = torch.sigmoid(cls_pred).max(dim=-1).values
                    labels_pred = torch.argmax(cls_pred, dim=-1)
                    # print(reg_pred)
                    boxes_encode = encoder2boxes(reg_pred, stride, (imagesize, imagesize))
                    count += H * W

                    boxes.append(boxes_encode.view(-1, 4))
                    labels.append(labels_pred.view(-1))
                    scores.append(scores_pred.view(-1))

                boxes = torch.cat(boxes)
                labels = torch.cat(labels)
                scores = torch.cat(scores)

                class_names = category_name[dataset]

                boxes_nms, scores_nms, labels_nms = [], [], []
                sum = 0
                NMS_threshold = 0.1
                for lab in range(0, num_class):
                    class_mask = labels == lab
                    score_mask = scores >= conf_thres
                    mask = class_mask & score_mask
                    if class_mask.sum() > 0:
                        indices = class_nms_indices(boxes[mask], scores[mask], iou_threshold=NMS_threshold)
                        boxes_nms.append(boxes[mask][indices])
                        scores_nms.append(scores[mask][indices])
                        labels_nms.append(labels[mask][indices])
                        sum += len(boxes_nms[-1])

                if len(boxes_nms) > 0:
                    boxes = torch.cat(boxes_nms)
                    scores = torch.cat(scores_nms)
                    labels = torch.cat(labels_nms)
                
                wh_ = boxes[:, 2:] - boxes[:, 0:2]

                for box, wh, sco, lab in zip(boxes, wh_, scores, labels):
                    temp_pred = {
                        "image_id": img_id,
                        "category_id": lab.item(),
                        "bbox": [box[0].item(), box[1].item(), wh[0].item(), wh[1].item()],
                        "score": sco.item()
                    }
                    predictions.append(temp_pred)
                img_id += 1
                
                boxes_nms = [b.cpu().detach().numpy() for b in boxes_nms]
                scores_nms = [s.cpu().detach().numpy() for s in scores_nms]
                labels_nms = [l.cpu().detach().numpy() for l in labels_nms]

                if batch_idx > 4:
                    continue
                if plot:
                    # print(boxes_nms)
                    visualize_detections(test_image.permute(1, 2, 0),
                                        0.5,
                                        target,
                                        boxes_nms,
                                        scores_nms, 
                                        labels_nms, 
                                        class_names,
                                        epoch,
                                        batch_idx,
                                        method,
                                        dataset
                                        )
        
        if len(predictions) == 0:
            index = {'precision':0, 'recall':0, 'F1':0, 'AP':0, 'mAP5095':0}

        else:
            coco_format = {
                "images": images_ann,
                "annotations": annotations,
                "categories": categories
            }
            coco_format = convert_to_python_type(coco_format)
            with open("annotations.json", "w") as f:
                json.dump(coco_format, f, indent=2)  # indent 使 JSON 更易读
                predictions = convert_to_python_type(predictions)
            with open("predictions.json", "w") as f:
                json.dump(predictions, f, indent=2)


            index = evaluate_coco_metrics("annotations.json", "predictions.json")
    
        print(index)
        return index


def calculate_detection_metrics(coco_eval, iou_thr=0.5):
    coco_eval.params.iouThrs = np.array([iou_thr])
    coco_eval.params.areaRng = [[0, 1e5**2]]  # 所有面积范围
    coco_eval.params.maxDets = [100]  # 限制最大检测数，PASCAL VOC通常每图检测数有限

    # 
    coco_eval.evaluate()

    # 
    TP, FP, FN = 0, 0, 0

    for eval_img in coco_eval.evalImgs:
        if eval_img is None:
            continue
        dt_matches = eval_img["dtMatches"][0]  # IoU阈值对应的匹配
        dt_ignore = eval_img["dtIgnore"][0]    # 忽略的检测
        gt_ignore = eval_img["gtIgnore"]       # 忽略的GT（e.g., difficult）
        gt_matches = eval_img["gtMatches"][0]  # GT的匹配情况

        for i, dt_match in enumerate(dt_matches):
            if dt_ignore[i]:  # 
                continue
            if dt_match > 0:  # 
                TP += 1
            else:  #
                FP += 1

        for i, gt_match in enumerate(gt_matches):
            if gt_ignore[i]:  
                continue
            if gt_match == 0: 
                FN += 1

    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return TP, FP, FN, precision, recall, f1


def evaluate_coco_metrics(gt_json_path, pred_json_path):
    with open(os.devnull, 'w') as devnull:
        original_stdout = sys.stdout
        sys.stdout = devnull
        try:
            coco_gt = COCO(gt_json_path)
            coco_dt = coco_gt.loadRes(pred_json_path)

            coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')

            TP, FP, FN, precision, recall, f1 = calculate_detection_metrics(coco_eval, iou_thr=0.5)

            coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')
            coco_eval.evaluate()
            coco_eval.accumulate()
            coco_eval.summarize()  # 打印标准输出（mAP 等）

            stats = coco_eval.stats

            index = {}

            index['precision'] = round(precision*100, 2)
            index['recall'] = round(recall*100, 2)
            index['F1'] = round(f1*100, 2)
            index['AP'] = round(stats[1]*100, 2)
            index['mAP5095'] = round(stats[0]*100, 2)
            return index
        finally:
            sys.stdout = original_stdout




def convert_to_python_type(obj):
    if isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_to_python_type(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_python_type(item) for item in obj]
    else:
        return obj


def class_nms_indices(boxes, scores, iou_threshold):
    indices = nms(boxes, scores, iou_threshold)
    return indices