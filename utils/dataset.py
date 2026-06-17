import os
import random

import cv2
import numpy as np
import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader, Dataset


DATASET_PATHS = {
    "PDT": {
        "train_images": "../dataset/PDT/PDT/LL/YOLO_txt/train/images/",
        "train_labels": "../dataset/PDT/PDT/LL/YOLO_txt/train/labels/",
        "val_images": "../dataset/PDT/PDT/LL/YOLO_txt/val/images/",
        "val_labels": "../dataset/PDT/PDT/LL/YOLO_txt/val/labels/",
    },
    "UCAS": {
        "train_images": "../dataset/UCAS_AOD/plane/train/images/",
        "train_labels": "../dataset/UCAS_AOD/plane/train/labels/",
        "val_images": "../dataset/UCAS_AOD/plane/val/images/",
        "val_labels": "../dataset/UCAS_AOD/plane/val/labels/",
    },
}


def random_flip(image, boxes):
    if random.random() < 0.5:
        image = np.fliplr(image)
        boxes[:, 0] = 1 - boxes[:, 0]
    if random.random() < 0.5:
        image = np.flipud(image)
        boxes[:, 1] = 1 - boxes[:, 1]
    return image, boxes


def reshape_transform(img_size):
    return T.Compose(
        [
            T.ToPILImage(),
            T.Resize([img_size, img_size]),
            T.ToTensor(),
        ]
    )


class YoloDetectionDataset(Dataset):
    def __init__(
        self, image_dir, annotations_path, img_size, transforms=None, is_train=True
    ):
        self.image_dir = image_dir
        self.annotations_path = annotations_path
        self.transforms = transforms
        self.is_train = is_train
        self.img_size = img_size
        self.annotations_list = []
        self.filename_list = []

        for annotation_name in sorted(os.listdir(annotations_path)):
            if not annotation_name.endswith(".txt"):
                continue
            image_name = annotation_name.replace(".txt", ".jpg")
            image = cv2.imread(os.path.join(image_dir, image_name))
            if image is None:
                raise FileNotFoundError(
                    f"Image not found: {os.path.join(image_dir, image_name)}"
                )

            self.filename_list.append(image_name)
            self.annotations_list.append(annotation_name)

    def __getitem__(self, idx):
        filename = self.filename_list[idx]
        image = cv2.imread(os.path.join(self.image_dir, filename))

        labels = []
        boxes = []
        annotation_file = os.path.join(
            self.annotations_path, self.annotations_list[idx]
        )
        with open(annotation_file, "r", encoding="utf-8") as file:
            for line in file:
                label, x_center, y_center, width, height = map(
                    float, line.strip().split()
                )
                labels.append(int(label))
                boxes.append([x_center, y_center, width, height])

        labels = torch.as_tensor(labels)
        boxes = torch.as_tensor(boxes)
        target = {"boxes": boxes, "labels": labels}

        if self.is_train:
            if boxes.shape[0] > 0:
                image, boxes = random_flip(image, boxes)
                target["boxes"] = boxes
            return self.transforms(image), target

        original_transform = reshape_transform(img_size=self.img_size)
        return (original_transform(image), self.transforms(image)), target

    def __len__(self):
        return len(self.annotations_list)


def collate_fn(batch):
    images, targets = zip(*batch)
    images = torch.stack(images, 0)

    labels = [item["labels"] for item in targets]
    boxes = [item["boxes"] for item in targets]
    max_gt = max((len(batch_labels) for batch_labels in labels), default=0)

    padded_boxes = torch.zeros([len(labels), max_gt, 4], device=images.device)
    padded_labels = torch.full([len(labels), max_gt], -1, device=images.device)

    for idx, box in enumerate(boxes):
        if len(box) > 0:
            padded_boxes[idx, : len(box)] = box

    for idx, batch_labels in enumerate(labels):
        if len(batch_labels) > 0:
            padded_labels[idx, : len(batch_labels)] = batch_labels

    return images, {"boxes": padded_boxes, "labels": padded_labels}


def collate_fn_val(batch):
    images, targets = zip(*batch)
    resized_images = [item[0] for item in images]
    original_images = [item[1] for item in images]

    resized_images = torch.stack(resized_images, 0)
    original_images = torch.stack(original_images, 0)

    return (resized_images, original_images), targets


def get_transform(img_size):
    norm_mean = [0.485, 0.456, 0.406]
    norm_std = [0.229, 0.224, 0.225]
    return T.Compose(
        [
            T.ToPILImage(),
            T.Resize([img_size, img_size]),
            T.ToTensor(),
            T.Normalize(norm_mean, norm_std),
        ]
    )


def getDataset(dataset, size, batchsize, testbatchsize=8):
    if dataset not in DATASET_PATHS:
        raise ValueError(f"Unsupported dataset: {dataset}")

    paths = DATASET_PATHS[dataset]
    train_dataset = YoloDetectionDataset(
        image_dir=paths["train_images"],
        annotations_path=paths["train_labels"],
        img_size=size,
        transforms=get_transform(img_size=size),
        is_train=True,
    )
    test_dataset = YoloDetectionDataset(
        image_dir=paths["val_images"],
        annotations_path=paths["val_labels"],
        img_size=size,
        transforms=get_transform(img_size=size),
        is_train=False,
    )

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batchsize,
        shuffle=True,
        num_workers=8,
        pin_memory=True,
        collate_fn=collate_fn,
    )
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=testbatchsize,
        shuffle=True,
        num_workers=8,
        pin_memory=True,
        collate_fn=collate_fn_val,
    )

    return train_loader, test_loader
