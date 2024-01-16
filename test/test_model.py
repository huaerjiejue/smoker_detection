#!/usr/bin/env python
# _*_ coding: utf-8 _*_
# @Time : 2023/12/17 22:32
# @Author : ZhangKuo
import torch
import torch.nn as nn
from utils.model import SmokeModel
import pytest
import cv2
from PIL import Image
import matplotlib.pyplot as plt


class TestModel:
    @pytest.fixture(scope="function")
    def pretreatment(self):
        pretreatment = SmokeModel(
            train_data_path="E:\\PycharmProjects\\smoker_detection\\data\\Training\\Training",
            valida_data_path="E:\\PycharmProjects\\smoker_detection\\data\\Validation\\Validation",
            batch_size=16,
            num_workers=4,
            lr=0.001,
            num_epochs=10,
        )
        return pretreatment

    def test_handle_data(self, pretreatment):
        pretreatment.handle_data()
        print("\ntrain_dataloader: ")
        print(pretreatment.train_dataloader.dataset[0]["img"].shape)
        print(len(pretreatment.train_dataloader.dataset))
        assert True
        print("valida_dataloader: ")
        print(pretreatment.valida_dataloader.dataset[0]["img"].shape)
        print(len(pretreatment.valida_dataloader.dataset))
        assert True
        # 可视化
        fig, axes = plt.subplots(nrows=5, ncols=2, figsize=(10, 10))
        for i in range(5):
            img = pretreatment.train_dataloader.dataset[i]["img"]
            img = img.permute(1, 2, 0)
            img = img.numpy()
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            axes[i][0].imshow(img)
            axes[i][0].set_title("train image")
            img = pretreatment.valida_dataloader.dataset[i]["img"]
            img = img.permute(1, 2, 0)
            img = img.numpy()
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            axes[i][1].imshow(img)
            axes[i][1].set_title("valida image")
        plt.show()
        assert True

    def test_handle_model(self, pretreatment):
        pretreatment.handle_data()
        pretreatment.handle_model()
        assert pretreatment.model.head.out_features == 2
