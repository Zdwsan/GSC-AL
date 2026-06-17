import torch
import math
from torchvision import transforms
import random

__all__ = ['ImageTransform']


class ImageTransform(dict):
    def __init__(self, size):
        super().__init__({
            'train': self.build_train_transform(size),
            'val': self.build_val_transform(size)
        })
        # self.size = size

    def build_train_transform(self, size):
        from timm.data import create_transform
        t = create_transform(
            input_size=size,
            is_training=True,
            color_jitter=0.4,
            mean=self.mean_std['mean'],
            std=self.mean_std['std'],
        )

        return t
    
    def build_val_transform(self, size):
        return transforms.Compose([
            transforms.Resize(int(math.ceil(size / 0.875))),
            transforms.CenterCrop(size),
            transforms.ToTensor(),
            transforms.Normalize(**self.mean_std)
        ])

    @property
    def mean_std(self):
        if True:  # MCU side model
            print('Using MCU transform (leading to range -128, 127)')
            return {'mean': [0.5, 0.5, 0.5], 'std': [0.5, 0.5, 0.5]}
        else:
            return configs.data_provider.get('mean_std',
                                             {'mean': [0.5, 0.5, 0.5],
                                              'std': [0.5, 0.5, 0.5]})
