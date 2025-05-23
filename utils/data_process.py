#!/usr/bin/env python
# _*_ coding: utf-8 _*_
# @Time : 2023/12/15 18:12
# @Author : ZhangKuo
import os
import random
from glob import glob
import cv2
from PIL import Image

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
from torch.utils.data import Dataset, DataLoader

"""
TODO:
  1. 读取数据
  2. 对数据进行预处理，包括数据增强和标准化
  3. 创建一个数据加载器，用于训练和测试
  4. 数据可视化
"""


class DataProcess:
    def __init__(self, data_path):
        self._data_path = data_path
        """
        path: 图片路径
        label: 图片标签
        class: 图片类别
        """
        self.df = pd.DataFrame({"path": [], "label": [], "class": []})
        self._labels = ["notsmoking", "smoking"]

    def get_data(self):
        """
        读取数据
        :return:
        """
        file_names = glob(self._data_path + "/*.jpg")
        for file_name in file_names:
            label = os.path.splitext(file_name)[0].split("\\")[-1]
            label = label.split("_")[0].lower()
            # label = str(label).lower()
            if label == self._labels[0]:
                class_ = 0.0
                new_df = pd.DataFrame(
                    {"path": [file_name], "label": [label], "class": class_}, index=[1]
                )
            elif label == self._labels[1]:
                class_ = 1.0
                new_df = pd.DataFrame(
                    {"path": [file_name], "label": [label], "class": class_}, index=[1]
                )
            else:
                raise ValueError("label error: " + label)
            self.df = pd.concat([self.df, new_df], axis=0, ignore_index=True)
        self.df["path"] = self.df["path"].astype(str)
        self.df["label"] = self.df["label"].astype(str)
        self.df["class"] = self.df["class"].astype(int)

    def draw_data(self):
        """
        数据可视化
        :return:
        """
        plt.figure(figsize=(10, 6))
        sns.countplot(x="label", hue="class", data=self.df, palette="Set1")
        plt.title("Smoking and Not Smoking")
        plt.xlabel("Label")
        plt.ylabel("Count")

        plt.legend(["Not Smoking", "Smoking"])

        # 在每个柱子上添加具体的数字
        for x, y in enumerate(self.df["class"].value_counts().values):
            plt.text(x, y + 1, "%s" % y, ha="center", va="bottom")

        plt.show()

    def draw_pic(self):
        """
        显示图片
        :return:
        """
        show_num = 15
        idxs = random.sample(range(len(self.df)), show_num)
        flg, axs = plt.subplots(3, 5, figsize=(25, 15))
        for i, idx in enumerate(idxs):
            img = plt.imread(self.df["path"][idx])
            label = self.df["label"][idx]
            idx_x, idx_y = i // 5, i % 5
            axs[idx_x, idx_y].imshow(img)
            axs[idx_x, idx_y].set_title(label)
        plt.show()

    def get_num_class(self):
        return len(self.df["class"].unique())


class SmokeDataset(Dataset):
    def __init__(self, data_path, transform=None):
        self._data_path = data_path
        self._transform = transform
        self._data_process = DataProcess(data_path=self._data_path)
        self._data_process.get_data()
        # self._data_process.draw_data()
        # self._data_process.draw_pic()

    def __len__(self):
        return len(self._data_process.df)

    def __getitem__(self, idx):
        img_path = self._data_process.df["path"][idx]
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        # label = self._data_process.df["class"][idx]
        # label为一个one-hot向量
        label = np.zeros(2)
        label[self._data_process.df["class"][idx]] = 1
        sample = {"img": img, "label": label}
        if self._transform:
            sample["img"] = self._transform(sample["img"])
        return sample

    def get_loader(self, batch_size, shuffle, num_workers):
        return DataLoader(
            dataset=self,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
        )

    def get_num_class(self):
        return self._data_process.get_num_class()
