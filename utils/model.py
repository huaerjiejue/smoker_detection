#!/usr/bin/env python
# _*_ coding: utf-8 _*_
# @Time : 2023/12/17 22:22
# @Author : ZhangKuo
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from sklearn.metrics import classification_report
from torch.utils.tensorboard import SummaryWriter
from torchvision.models import Swin_V2_S_Weights
from tqdm import tqdm

from utils.data_process import SmokeDataset
from utils.transform import Transform


class SmokeModel:
    def __init__(
        self, train_data_path, valida_data_path, batch_size, num_workers, lr, num_epochs
    ):
        self._train_data_path = train_data_path
        self._valida_data_path = valida_data_path
        self._batch_size = batch_size
        self._num_workers = num_workers
        self._lr = lr
        self._num_epochs = num_epochs
        self._device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = models.swin_v2_s(weights=Swin_V2_S_Weights.IMAGENET1K_V1)
        # swin_v2_s 推荐transform是 mean=[0.485, 0.456, 0.406]，std=[0.229, 0.224, 0.225]
        self.model.to(self._device)
        self._criterion = nn.CrossEntropyLoss()
        self._optimizer = optim.AdamW(self.model.parameters(), lr=self._lr)
        self._scheduler = optim.lr_scheduler.StepLR(
            self._optimizer, step_size=7, gamma=0.1
        )
        self._mean = [0.485, 0.456, 0.406]
        self._std = [0.229, 0.224, 0.225]
        self.log_dir = "E:/smoker_detection/logs"
        self.writer = SummaryWriter(log_dir=self.log_dir)
        self.train_dataloader = None
        self.valida_dataloader = None
        self._valida_acc = 0.0
        self._num_class = 0

    def handle_data(self):
        train_transform = Transform(self._mean, self._std).train_transform()
        test_transform = Transform(self._mean, self._std).test_transform()
        train_data = SmokeDataset(
            data_path=self._train_data_path, transform=train_transform
        )
        valida_data = SmokeDataset(
            data_path=self._valida_data_path, transform=test_transform
        )
        if train_data.get_num_class() != valida_data.get_num_class():
            raise ValueError("train_data and valida_data have different num_class")
        self._num_class = train_data.get_num_class()
        self.train_dataloader = train_data.get_loader(
            batch_size=self._batch_size, shuffle=True, num_workers=self._num_workers
        )
        self.valida_dataloader = valida_data.get_loader(
            batch_size=self._batch_size, shuffle=True, num_workers=self._num_workers
        )

    def handle_model(self):
        # 在使用这个函数之前，需要先处理数据，不然num_class不会被赋值
        self.model.head = nn.Linear(
            in_features=self.model.head.in_features,
            out_features=self._num_class,
            bias=True,
        )

    def train(self):
        for epoch in tqdm(range(self._num_epochs)):
            # self._scheduler.step()
            self.model.train()
            running_loss = 0.0
            for idx, data in enumerate(self.train_dataloader):
                inputs, labels = data["img"].to(self._device), data["label"].to(
                    self._device
                )
                self._optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self._criterion(outputs, labels)
                loss.backward()
                self._optimizer.step()
                running_loss += loss.item()
                if idx % 10 == 9:
                    self.writer.add_scalar(
                        "train_loss",
                        running_loss / 10,
                        epoch * len(self.train_dataloader) + idx,
                    )
                    running_loss = 0.0
            self._scheduler.step()
            self.valida()
            self.writer.add_scalar("valida_acc", self._valida_acc, epoch)
            print(f"train_loss: {running_loss / 10}, valida_acc: {self._valida_acc}")

    def valida(self):
        self.model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for data in self.valida_dataloader:
                images, labels = data["img"].to(self._device), data["label"].to(
                    self._device
                )
                outputs = self.model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum()
        self._valida_acc = 100 * correct / total

    def show_result(self):
        # 使用 sklearn.metrics 中的 classification_report进行绘制
        self.model.eval()
        y_true = []
        y_pred = []
        with torch.no_grad():
            for data in self.valida_dataloader:
                images, labels = data["img"].to(self._device), data["label"].to(
                    self._device
                )
                outputs = self.model(images)
                _, predicted = torch.max(outputs.data, 1)
                y_true += labels.cpu().numpy().tolist()
                y_pred += predicted.cpu().numpy().tolist()
        target_names = ["notsmoking", "smoking"]
        print(classification_report(y_true, y_pred, target_names=target_names))


if __name__ == "__main__":
    model = SmokeModel(
        train_data_path="E:\\PycharmProjects\\smoker_detection\\data\\Training\\Training",
        valida_data_path="E:\\PycharmProjects\\smoker_detection\\data\\Validation\\Validation",
        batch_size=16,
        num_workers=4,
        lr=0.001,
        num_epochs=10,
    )
    model.handle_data()
    model.handle_model()
    model.train()
    model.show_result()
