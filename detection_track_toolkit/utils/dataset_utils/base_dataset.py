# -*- coding: utf-8 -*-
# @Time    : 2026/9/10 09:06
# @Author  : DaiPuWei
# @Email   : daipuwei@qq.com
# @File    : base_dataset.py
# @Software: PyCharm

"""
    这是定义数据集加载器基类的脚本。

    模块内容：
        1. Dataset：数据集加载器基类，提供批量迭代加载的通用实现：
           - 实现迭代协议（__len__/__iter__/__next__）：按batchsize切分meta_data，
             逐批惰性读取图像（取批时才cv2.imread），尾批不满batchsize时照常返回
             （batchsize大于图像数量时整集作为一批返回），迭代完一轮后游标
             自动复位、可再次迭代；
           - get_batch_data(start, end)：批量加载钩子（子类可覆写以加载标注等
             附加数据），图像与元数据成对返回；
           - 校验：batchsize必须为正整数；批量图像尺寸不一致时明确报错
             （尺寸不一的数据集请使用batchsize=1）；图像读取失败立即报错
             （而不是返回None后在远处才崩）。
        子类：utils.dataset_utils.voc.VOCDataset、utils.dataset_utils.coco.COCODataset，
        负责数据集格式解析与元数据构造，批量加载复用基类实现。
"""

import cv2
import numpy as np

class Dataset(object):

    def __init__(self,meta_data,batchsize=1):
        """
        这是数据集加载器的初始化函数
        Args:
            meta_data: 数据集元数据（list或ndarray，每个元素为含image_path键的字典）
            batchsize: 小批量数据规模，默认为1
        """
        # 初始化相关参数
        if isinstance(meta_data,list):
            meta_data = np.array(meta_data)
        self.meta_data = meta_data
        if batchsize <= 0:
            raise ValueError("batchsize必须为正整数，实际为{0}".format(batchsize))
        self.batchsize = batchsize
        self.start = 0
        self.end = 0
        self.image_num = len(self.meta_data)

    def __len__(self):
        """
        这是获取数据集批量数量的函数（尾批不满batchsize也计为一批）
        Returns:
            批量数量
        """
        size = self.image_num // self.batchsize
        if self.image_num % self.batchsize != 0:
            size += 1
        return size

    def get_batch_data(self,start,end):
        """
        这是加载一个batchsize数据的函数（批量加载钩子，子类可覆写
        以加载标注等附加数据）
        Args:
            start: 开始索引
            end: 结束索引
        Returns:
            batch_images: 批量图像数组（同尺寸校验通过后拼接）
            batch_meta_data: 批量元数据数组
        """
        batch_meta_data = self.meta_data[start:end]
        batch_images = []
        for meta_data in batch_meta_data:
            image = cv2.imread(meta_data['image_path'])
            if image is None:
                raise IOError("图像读取失败：{0}".format(meta_data['image_path']))
            batch_images.append(image)
        # 同尺寸校验：尺寸不一的图像集请使用batchsize=1
        shapes = [image.shape for image in batch_images]
        if len(set(shapes)) > 1:
            raise ValueError("批量图像尺寸不一致{0}：图像尺寸不一时请使用batchsize=1"
                             .format(shapes))
        batch_images = np.array(batch_images)
        return batch_images,batch_meta_data

    def __next__(self):
        # 迭代结束判断：当前游标已达末尾则复位游标并结束本轮迭代
        if self.start >= self.image_num:
            self.start = 0
            self.end = 0
            raise StopIteration
        # 取当前批次（尾批不满batchsize照常返回）
        self.end = min(self.start+self.batchsize,self.image_num)
        batch_images,batch_meta_data = self.get_batch_data(self.start,self.end)
        self.start = self.end
        return batch_images,batch_meta_data

    def __iter__(self):
        return self

    def get_image_num(self):
        """
        这是获取数据集图像数量的函数
        Returns:
            图像数量
        """
        return self.image_num
