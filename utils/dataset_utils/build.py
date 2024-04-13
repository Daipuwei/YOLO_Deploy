# -*- coding: utf-8 -*-
# @Time    : 2024/4/13 下午10:17
# @Author  : DaiPuWei
# @Email   : daipuwei@qq.com
# @File    : build.py
# @Software: PyCharm


"""
    这是定义数据集注册器的脚本
"""

from utils import Registry

DATASET_REGISTRY = Registry("DATASET")
DATASET_REGISTRY.__doc__ = """这是数据集的注册器，用于自动生成不同数据集类，例如VOC、COCO"""

def build_dataset(dataset_type,dataset_dir,batchsize=1,mode='train'):
    """
    这是搭建数据集加载器的函数
    Args:
        dataset_type: 数据集类型
        dataset_dir: 数据集地址
        batchsize: 小批量数据规模，默认为1
        mode: 子集类型，默认为‘train’
    Returns:
    """
    return DATASET_REGISTRY.get(dataset_type)(dataset_dir,batchsize,mode)
