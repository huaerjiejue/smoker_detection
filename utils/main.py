#!/usr/bin/env python
# _*_ coding: utf-8 _*_
# @Time : 2024/1/16 19:50
# @Author : ZhangKuo
import argparse

import numpy as np
import torch
from PIL import Image

from utils.model import SmokeModel


def train(args):
    pretreatment = SmokeModel(
        train_data_path=args.train_data_path,
        valida_data_path=args.valida_data_path,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        lr=args.lr,
        num_epochs=args.num_epochs,
    )
    pretreatment.handle_data()
    pretreatment.handle_model()
    pretreatment.train()


def show(args):
    pretreatment = SmokeModel(
        train_data_path=args.train_data_path,
        valida_data_path=args.valida_data_path,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        lr=args.lr,
        num_epochs=args.num_epochs,
    )
    pretreatment.handle_data()
    pretreatment.handle_model()
    pretreatment.show_result()


def use_model(args):
    model = torch.load(args.model_path)
    model.eval()
    model.to(torch.device("cuda:0"))
    img = Image.open(args.img_path)
    img = img.resize((224, 224))
    img = torch.from_numpy(np.array(img))
    img = img.permute(2, 0, 1)
    img = img.unsqueeze(0)
    img = img.float()
    img = img.to(torch.device("cuda:0"))
    output = model(img)
    print(output)
    _, predicted = torch.max(output, 1)
    print(predicted)
    if predicted == 0:
        print("notsmoking")
    elif predicted == 1:
        print("smoking")
    else:
        raise ValueError("predicted error: " + predicted)


def main():
    parser = argparse.ArgumentParser(description="smoke detection")
    parser.add_argument(
        "--train_data_path",
        type=str,
        default="E:\\PycharmProjects\\smoker_detection\\data\\Training\\Training",
        help="train data path",
    )
    parser.add_argument(
        "--valida_data_path",
        type=str,
        default="E:\\PycharmProjects\\smoker_detection\\data\\Validation\\Validation",
        help="valida data path",
    )
    parser.add_argument(
        "--batch_size", type=int, default=16, help="batch size for training"
    )
    parser.add_argument(
        "--num_workers", type=int, default=4, help="num workers for training"
    )
    parser.add_argument("--lr", type=float, default=0.001, help="learning rate")
    parser.add_argument(
        "--num_epochs", type=int, default=10, help="num epochs for training"
    )
    parser.add_argument(
        "--mode", type=str, default="train", help="train or show result"
    )
    parser.add_argument(
        "--model_path", type=str, default="model.pth", help="model path"
    )
    parser.add_argument("--img_path", type=str, default="test.jpg", help="img path")
    args = parser.parse_args()
    if args.mode == "train":
        train(args)
    elif args.mode == "show":
        show(args)
    elif args.mode == "use_model":
        use_model(args)
    else:
        raise ValueError("mode error: " + args.mode)
