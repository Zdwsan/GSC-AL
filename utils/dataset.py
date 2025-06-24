import torchvision.transforms as T
import json
import os
import random
import re
import cv2
from torch.utils.data import Dataset, DataLoader
from pycocotools.coco import COCO
from compute.function import *
from PIL import Image
from utils.categories import *


def random_flip(im, boxes):
    if random.random() < 0.5:
        im = np.fliplr(im) 
        boxes[:, 0]  = 1 - boxes[:, 0] 
    if random.random() < 0.5:
        im = np.flipud(im)
        boxes[:, 1] = 1 - boxes[:, 1]
    return im, boxes

        
def reshape_transform(img_size):
    transforms = [T.ToPILImage()]
    transforms.append(T.Resize([img_size, img_size]))
    transforms.append(T.ToTensor())
    return T.Compose(transforms)


class myDatasets(Dataset):
    def __init__(self, image_dir, annotations_path, img_size, transforms=None, isTrain=True):
        self.image_dir = image_dir
        self.annotations_path = annotations_path
        self.transforms = transforms
        self.data = []
        self.images = []
        self.scale_w_h = []
        self.isTrain = isTrain
        self.img_size = img_size
        self.annotationsList = []
        self.filenameList = []
        annoList = os.listdir(annotations_path)
        for anno_name in annoList:
            img = cv2.imread(image_dir + anno_name.replace('.txt', '.jpg')) 
            H, W, C = img.shape
            self.scale_w_h.append([img_size/W, img_size/H])
            self.filenameList.append(anno_name.replace('.txt', '.jpg'))
            self.annotationsList.append(anno_name)

    def __getitem__(self, idx):
        filename = self.filenameList[idx]
        img = cv2.imread(self.image_dir + filename)

        annotationsFile = self.annotationsList[idx]
        labels = []
        boxes = []
        with open(self.annotations_path + annotationsFile, 'r') as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip()
                line = line.split()
                l, xc, yc, w, h = map(float, line)
                labels.append(int(l))
                boxes.append([ xc, yc, w, h])
            labels = torch.as_tensor(labels)
            boxes = torch.as_tensor(boxes)
        if self.isTrain:
            if boxes.shape[0] > 0:
                img, boxes = random_flip(img, boxes)
            target = {'boxes': boxes, 'labels': labels}
            img_trans = self.transforms(img)
            return img_trans, target
        else: 
            target = {'boxes': boxes, 'labels': labels}
            reshapeForm = reshape_transform(img_size=self.img_size)
            return (reshapeForm(img), self.transforms(img)), target

    
    def __len__(self):
        return len(self.annotationsList)

def collate_fn(batch):
    imgs, targets = zip(*batch)
    imgs = torch.stack(imgs, 0)  # [B,C,H,W]

    labels = [item['labels'] for item in targets]
    boxes = [item['boxes'] for item in targets]

    max_gt = 0
    for b_gt in labels:
        if max_gt < len(b_gt):
            max_gt = len(b_gt)

    padded_boxes = torch.zeros([len(labels), max_gt, 4], device=imgs.device)
    
    for i , box in enumerate(boxes):
        if len(box) > 0:
            padded_boxes[i, :len(box)] = box

    padded_labels = torch.zeros([len(labels), max_gt], device=imgs.device)
    
    for i, b_gt in enumerate(labels):
        if len(b_gt) > 0:
            padded_labels[i, :len(b_gt)] = b_gt
            padded_labels[i, len(b_gt):] = -1
    padded_targets = {'boxes': padded_boxes, 'labels':padded_labels}            
    return imgs, padded_targets

def collate_fn_val(batch):
    images, targets = zip(*batch)
    imgs = [ig[0] for ig in images]
    o_imgs = [ig[1] for ig in images]

    imgs = torch.stack(imgs, 0)  # [B,C,H,W]
    o_imgs = torch.stack(o_imgs, 0)

    return (imgs, o_imgs), targets

def get_transform(img_size):
    norm_mean = [0.485, 0.456, 0.406]
    norm_std = [0.229, 0.224, 0.225]
    transforms = [T.ToPILImage()]
    transforms.append(T.Resize([img_size, img_size]))
    transforms.append(T.ToTensor())
    transforms.append(T.Normalize(norm_mean, norm_std))
    return T.Compose(transforms)

def getDataset(dataset, size, batchsize, testbatchsize=8):
    if dataset == 'PDT':
        train_dataset = myDatasets(
            image_dir = '../dataset/PDT/PDT/LL/YOLO_txt/train/images/',
            annotations_path = '../dataset/PDT/PDT/LL/YOLO_txt/train/labels/',
            img_size=size,
            transforms=get_transform(img_size=size),
            isTrain=True
        )

        test_dataset = myDatasets(
            image_dir = '../dataset/PDT/PDT/LL/YOLO_txt/val/images/',
            annotations_path = '../dataset/PDT/PDT/LL/YOLO_txt/val/labels/',
            img_size=size,
            transforms=get_transform(img_size=size),
            isTrain=False
        )

    elif dataset == 'UCAS':
        train_dataset = myDatasets(
            image_dir = '../dataset/UCAS_AOD/plane/train/images/',
            annotations_path = '../dataset/UCAS_AOD/plane/train/labels/',
            img_size=size,
            transforms=get_transform(img_size=size),
            isTrain=True
        )

        test_dataset = myDatasets(
            image_dir = '../dataset/UCAS_AOD/plane/val/images/',
            annotations_path = '../dataset/UCAS_AOD/plane/val/labels/',
            img_size=size,
            transforms=get_transform(img_size=size),
            isTrain=False
        )

    train_loader = DataLoader(dataset=train_dataset, batch_size=batchsize, shuffle=True, num_workers=8, pin_memory=True, collate_fn=collate_fn)
    test_loader = DataLoader(dataset=test_dataset, batch_size=testbatchsize, shuffle=True, num_workers=8, pin_memory=True, collate_fn=collate_fn_val)

    return train_loader, test_loader



