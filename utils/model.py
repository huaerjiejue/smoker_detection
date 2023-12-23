#!/usr/bin/env python
# _*_ coding: utf-8 _*_
# @Time : 2023/12/17 22:22
# @Author : ZhangKuo
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from torchvision.models import Swin_V2_S_Weights
from utils.data_process import DataProcess
from utils.data_process import SmokeDataset
from utils.transform import Transform
from torch.utils.data import Dataset, DataLoader


class SmokeModel:
    def __init__(
        self, train_data_path, test_data_path, batch_size, num_workers, lr, num_epochs
    ):
        self._train_data_path = train_data_path
        self._test_data_path = test_data_path
        self._batch_size = batch_size
        self._num_workers = num_workers
        self._lr = lr
        self._num_epochs = num_epochs
        self._device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self._model = models.swin_v2_s(weights=Swin_V2_S_Weights.IMAGENET1K_V1)
        # swin_v2_s 推荐transform是 mean=[0.485, 0.456, 0.406]，std=[0.229, 0.224, 0.225]
        self._model.to(self._device)
        self._criterion = nn.CrossEntropyLoss()
        self._optimizer = optim.SGD(self._model.parameters(), lr=self._lr, momentum=0.9)
        self._scheduler = optim.lr_scheduler.StepLR(
            self._optimizer, step_size=7, gamma=0.1
        )
        self._mean = [0.485, 0.456, 0.406]
        self._std = [0.229, 0.224, 0.225]
        self._traint_dataloader = None
        self._test_dataloader = None
        self._num_class = DataProcess(self._train_data_path).get_num_class()

    def _handle_data(self):
        train_transform = Transform(self._mean, self._std).train_transform()
        test_transform = Transform(self._mean, self._std).test_transform()
        train_data = SmokeDataset(
            data_path=self._train_data_path, transform=train_transform
        )
        test_data = SmokeDataset(
            data_path=self._test_data_path, transform=test_transform
        )
        self._traint_dataloader = SmokeDataset.get_loader(
           batch_size=self._batch_size,
           num_workers=self._num_workers,
           shuffle=True,
        )
        self._test_dataloader = SmokeDataset.get_loader(
            batch_size=self._batch_size,
            num_workers=self._num_workers,
            shuffle=False,
        )

    # 模型的框架需要进行更改
    def _handle_model(self):
        pass

    def train(self):
        pass

    def valida(self):
        pass

    def show_result(self):
        pass

    def show_tensorboard(self):
        pass
