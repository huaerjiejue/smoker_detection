#!/usr/bin/env python
# _*_ coding: utf-8 _*_
# @Time : 2023/12/17 15:47
# @Author : ZhangKuo
import pytest
from utils.transform import Transform
import matplotlib.pyplot as plt
from utils.data_process import DataProcess
from random import randint
from PIL import Image
import cv2


class TestTransform:
    @pytest.fixture(scope="function")
    def transform(self):
        transform = Transform(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], resize=256)
        return transform

    @pytest.fixture(scope="function")
    def data_process_train(self):
        data_process = DataProcess(
            data_path="E:\\PycharmProjects\\smoker_detection\\data\\Training\\Training"
        )
        return data_process

    @pytest.fixture(scope="function")
    def data_process_test(self):
        data_process = DataProcess(
            data_path="E:\\PycharmProjects\\smoker_detection\\data\\Testing\\Testing"
        )
        return data_process

    def test_train_transform(self, transform, data_process_train):
        train_transform = transform.train_transform()
        data_process_train.get_data()
        # 创建对比图像，有两列，对比前后的图像
        fig, axes = plt.subplots(nrows=5, ncols=2, figsize=(10, 10))
        # 随机选取5张图片
        for i in range(5):
            # 随机选取一张图片
            index = randint(0, len(data_process_train._df))
            # 读取图片
            img = cv2.imread(data_process_train._df.iloc[index]["path"])
            # 对图片进行数据增强
            ori_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(ori_img)
            img = train_transform(img)
            # 对比图像
            axes[i][0].imshow(ori_img)
            axes[i][0].set_title("original image")
            axes[i][1].imshow(img.permute(1, 2, 0))
            axes[i][1].set_title("transformed image")
        plt.show()

        assert True

    def test_test_transform(self, transform, data_process_test):
        test_transform = transform.test_transform()
        data_process_test.get_data()
        # 创建对比图像，有两列，对比前后的图像
        fig, axes = plt.subplots(nrows=5, ncols=2, figsize=(10, 10))
        # 随机选取5张图片
        for i in range(5):
            # 随机选取一张图片
            index = randint(0, len(data_process_test._df))
            # 读取图片
            img = cv2.imread(data_process_test._df.iloc[index]["path"])
            # 对图片进行数据增强
            ori_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(ori_img)
            img = test_transform(img)
            # 对比图像
            axes[i][0].imshow(ori_img)
            axes[i][0].set_title("original image")
            axes[i][1].imshow(img.permute(1, 2, 0))
            axes[i][1].set_title("transformed image")
        plt.show()

        assert True
