#!/usr/bin/env python
# _*_ coding: utf-8 _*_
# @Time : 2023/12/16 17:26
# @Author : ZhangKuo
import pytest

from utils.data_process import DataProcess


class TestDataProcess:
    @pytest.fixture(scope="function")
    def data_process(self):
        data_process = DataProcess(data_path="E:\\PycharmProjects\\smoker_detection\\data\\Testing\\Testing")
        return data_process

    def test_get_data(self, data_process):
        data_process.get_data()
        assert len(data_process.df) == 224
        assert 'notsmoking' in data_process.df['label'].values
        assert 'smoking' in data_process.df['label'].values
        print(data_process.df.head())
        assert True

    @pytest.fixture(scope="function")
    def get_data(self, data_process):
        data_process.get_data()
        return data_process

    def test_draw_data(self, get_data):
        get_data.draw_data()
        assert True

    def test_draw_pic(self, get_data):
        get_data.draw_pic()
        assert True

    def test_get_num_class(self, get_data):

        num_class = get_data.get_num_class()
        assert num_class == 2
        assert True