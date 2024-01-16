#!/usr/bin/env python
# _*_ coding: utf-8 _*_
# @Time : 2023/12/17 14:39
# @Author : ZhangKuo
import torch
from torchvision.transforms import v2


class Transform:
    def __init__(self, mean, std, resize=256):
        """

        :param mean: array-like, shape (C, H, W)
        :param std: array-like, shape (C, H, W)
        :param resize:  int, 图像大小
        """
        self._resize = resize
        self._mean = mean
        self._std = std

    def train_transform(self):
        """
        训练数据增强
        :return:
        """
        return v2.Compose(
            [
                v2.Resize(self._resize),
                v2.RandomResizedCrop(size=self._resize - 2, scale=(0.8, 1.0)),
                v2.RandomHorizontalFlip(p=0.5),
                v2.RandomVerticalFlip(p=0.5),
                v2.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.8, 1.2)),
                v2.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
                v2.RandomGrayscale(p=0.025),
                v2.RandomRotation(15),
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(self._mean, self._std),
            ]
        )

    def test_transform(self):
        """
        测试数据增强
        :return:
        """
        return v2.Compose(
            [
                v2.Resize(self._resize),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(self._mean, self._std),
            ]
        )
