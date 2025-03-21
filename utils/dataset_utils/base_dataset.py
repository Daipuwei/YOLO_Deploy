# -*- coding: utf-8 -*-
# @Time    : 2024/4/13 下午10:22
# @Author  : DaiPuWei
# @Email   : daipuwei@qq.com
# @File    : base_dataset.py
# @Software: PyCharm

"""
    这是定义抽象数据集加载器脚本
"""

import os
import cv2
import numpy as np

class Dataset(object):

    def __init__(self,meta_data,batchsize=1):
        """
        这是抽象数据集
        Args:
            meta_data: 数据集
            batchsize: 小批量数据规模，默认为1
        """
        # 初始化相关参数
        self.meta_data = meta_data
        self.batchsize = batchsize
        if isinstance(meta_data,list):
            self.meta_data = np.array(self.meta_data)
        self.start = 0
        self.end = 0
        self.image_num = len(meta_data)

    def __len__(self):
        size = self.image_num // self.batchsize
        if self.image_num % self.batchsize != 0:
            size += 1
        return size

    def get_batch_data(self,start,end):
        """
        这是加载一个batchsize数据的函数
        Args:
            start: 开始索引
            end: 结束索引
        Returns:
        """
        batch_meta_data = self.meta_data[start:end]
        batch_images = []
        for meta_data in batch_meta_data:
            image = cv2.imread(meta_data['image_path'])
            batch_images.append(image)
        batch_images = np.array(batch_images)
        return batch_images,batch_meta_data

    def __next__(self):
        self.end = min(self.start+self.batchsize,self.image_num)
        if self.end == self.image_num:
            self.start = 0
            self.end = 0
            raise StopIteration
        else:
            batch_images,batch_meta_data = self.get_batch_data(self.start,self.end)
            self.start = self.end
            return batch_images,batch_meta_data

    def __iter__(self):
        return self

    def get_image_num(self):
        return self.image_num