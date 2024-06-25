#  #!/usr/bin/env python
#  -*- coding:utf-8 -*-
#  Copyleft (C) 2024 proanimer, Inc. All Rights Reserved
#   author:proanimer
#   createTime:2024/6/19 下午8:48
#   lastModifiedTime:2024/6/19 下午8:48
#   file:dataloader.py
#   software: classicNets
#

from constants import Configure
from torch.utils.data import Dataset
import numpy as np
import random
from PIL import Image
import torch
from torchvision import transforms as T
from torchvision.transforms import functional as F
from torchvision import transforms
import os
import glob
from pathlib import Path
import cv2
from functools import partial


class Resize(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, image, target):
        image = F.resize(image, self.size)
        if target is not None:
            target = F.resize(target, self.size, interpolation=F.InterpolationMode.NEAREST)
        return image, target


class RandomHorizontalFlip(object):
    def __init__(self, flip_prob):
        self.flip_prob = flip_prob

    def __call__(self, image, target):
        if random.random() < self.flip_prob:
            image = F.hflip(image)
            if target is not None:
                target = F.hflip(target)
        return image, target


class RandomVerticalFlip(object):
    def __init__(self, flip_prob):
        self.flip_prob = flip_prob

    def __call__(self, image, target):
        if random.random() < self.flip_prob:
            image = F.vflip(image)
            if target is not None:
                target = F.vflip(target)
        return image, target


class RandomCrop(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, image, target):
        crop_params = T.RandomCrop.get_params(image, self.size)
        image = F.crop(image, *crop_params)
        if target is not None:
            target = F.crop(target, *crop_params)
        return image, target


class CenterCrop(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, image, target):
        image = F.center_crop(image, self.size)
        if target is not None:
            target = F.center_crop(target, self.size)
        return image, target


class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image, target):
        image = F.normalize(image, mean=self.mean, std=self.std)
        return image, target


class Pad(object):
    def __init__(self, padding_n, padding_fill_value=0, padding_fill_target_value=0):
        self.padding_n = padding_n
        self.padding_fill_value = padding_fill_value
        self.padding_fill_target_value = padding_fill_target_value

    def __call__(self, image, target):
        assert isinstance(image, (np.ndarray, torch.Tensor,
                                  Image.Image)), ("Image data must be either a numpy array, torch tensor, or a PIL "
                                                  "Image.")
        if isinstance(image, torch.Tensor) or isinstance(image, np.ndarray):
            height_padding = max(0, self.padding_n[0] - image.shape[0])
            width_padding = max(0, self.padding_n[1] - image.shape[1])
            image = np.pad(target, ((0, height_padding), (0, width_padding), (0, 0)), "constant",
                           constant_values=self.padding_fill_value)
        else:
            H, W = np.array(image).shape[:2]
            height_padding = max(0, self.padding_n[0] - H)
            width_padding = max(0, self.padding_n[1] - W)
            padding = [0, 0, width_padding, height_padding]
            image = F.pad(image, padding, self.padding_fill_target_value)
        if target is not None:
            if isinstance(target, torch.Tensor) or isinstance(target, np.ndarray):
                target = np.pad(target, ((0, height_padding), (0, width_padding), (0, 0)), "constant",
                                constant_values=self.padding_fill_value)
            else:
                H, W = np.array(target).shape[:2]
                height_padding = max(0, self.padding_n[0] - H)
                width_padding = max(0, self.padding_n[1] - W)
                padding = [0, 0, width_padding, height_padding]
                target = F.pad(target, padding, self.padding_fill_target_value)
        return image, target


class RandomRotation(object):
    def __init__(self, degrees=45, rotate_prob=.3):
        degrees = abs(degrees)
        self.degrees = random.randint(-degrees, degrees)
        self.rotate_prob = rotate_prob

    def __call__(self, image, target):
        if random.random() < self.rotate_prob:
            image = F.rotate(image, self.degrees)
            if target is not None:
                target = F.rotate(target, self.degrees)
        return image, target


class ToTensor(object):
    def __call__(self, image, target):
        image = F.to_tensor(image)
        if target is not None:
            target = torch.as_tensor(np.array(target), dtype=torch.int64)
        return image, target


class RandomGaussianBlur(object):
    def __init__(self, kernel_size=(5, 5), sigma=(0.1, 1.9), gaussian_prob=0.5):
        self.kernel_size = kernel_size
        self.sigma = sigma
        self.gaussian_prob = gaussian_prob

    def __call__(self, img, mask):
        if random.random() < self.gaussian_prob:
            sigma = random.uniform(self.sigma[0], self.sigma[1])
            return F.gaussian_blur(img, self.kernel_size, [sigma, sigma]), mask
        return img, mask


class ToPILImage(object):
    def __call__(self, image_array, target):
        print(image_array.shape)
        image = F.to_pil_image(image_array)
        if target is not None:
            target = F.to_pil_image(target)
        return image, target


class ColorJitter(object):
    """
    8 bit图颜色调整
    """

    def __init__(self, jitter_prob=.25):
        self.jitter_prob = jitter_prob

    def __call__(self, image, target):
        if random.random() < self.jitter_prob:
            # 彩色图可以进行以下数据增强，参数不太好调整
            image = F.adjust_gamma(image, gamma=random.uniform(0.8, 1.2))
            image = F.adjust_contrast(
                image, contrast_factor=random.uniform(0.8, 1.2))
            image = F.adjust_brightness(
                image, brightness_factor=random.uniform(0.8, 1.2))
            image = F.adjust_saturation(
                image, saturation_factor=random.uniform(0.8, 1.2))
            image = F.adjust_hue(image, hue_factor=random.uniform(-0.2, 0.2))
        return image, target


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, mask):
        for t in self.transforms:
            image, mask = t(image, mask)
        return image, mask


# 定义图像增强的Compose函数
def get_train_transforms(resize=False):
    compose_ops = [
        Pad((Configure.PADDING, Configure.PADDING)),
        RandomCrop((Configure.CROP_SIZE, Configure.CROP_SIZE)),
        RandomHorizontalFlip(flip_prob=Configure.FLIP_PROB),
        RandomVerticalFlip(flip_prob=Configure.FLIP_PROB),
        RandomRotation(degrees=45, rotate_prob=.3),
        ColorJitter(jitter_prob=.25),
        RandomGaussianBlur(kernel_size=(3, 3), sigma=(0.1, 1.8)),
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]
    if resize:
        compose_ops.insert(0, Resize(Configure.IMG_SIZE))
    transform = Compose(compose_ops)

    return transform


# 定义图像增强的Compose函数
def get_test_transforms(crop=False):
    compose_ops = [
        # Pad((Configure.PADDING, Configure.PADDING)), #由于在crop_image时 我创建(crop_size,crop_size)的小图像,做了类似padding操作,这里就不需要了
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]
    if crop:
        compose_ops.insert(0, Pad((Configure.PADDING, Configure.PADDING)))
    transform = Compose(compose_ops)
    return transform


class GarbageData(Dataset):
    def __init__(self, data_dir, transform=None):
        super().__init__()

        self.data_dir = data_dir

        self.transform = transform
        self.image_files, self.mask_files = self._load_files()

    def _load_files(self):
        image_files = []
        mask_files = []

        # 使用 glob 模块获取所有图像和掩码文件的路径
        image_paths = glob.glob(os.path.join(self.data_dir, '**/*.jpg'), recursive=True)
        mask_paths = glob.glob(os.path.join(self.data_dir, '**/*.png'), recursive=True)

        # 遍历图像文件路径
        for image_path in image_paths:
            # 构建对应的掩码文件路径
            mask_path = image_path.replace('.jpg', '.png')

            # 如果对应的掩码文件存在，则添加到列表中
            if mask_path in mask_paths:
                image_files.append(image_path)
                mask_files.append(mask_path)

        return image_files, mask_files

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, index):
        image_name = self.image_files[index]
        mask_name = self.mask_files[index]
        assert Path(image_name).stem == Path(mask_name).stem, "名称相同"
        image_path = os.path.join(self.data_dir, image_name)
        mask_path = os.path.join(self.data_dir, mask_name.replace('.jpg', '.png'))

        image = Image.open(image_path)
        mask = Image.open(mask_path)
        assert np.array(image).shape[0] == np.array(mask).shape[0] and np.array(image).shape[1] == np.array(mask).shape[
            1], "图片与mask尺寸不一致."
        if self.transform is not None:
            image, mask = self.transform(image, mask)

        return image, mask
