# -*- coding: utf-8 -*-
# @Time    : 2024/4/13 下午10:16
# @Author  : DaiPuWei
# @Email   : daipuwei@qq.com
# @File    : __init__.py.py
# @Software: PyCharm

from .build import DATASET_REGISTRY
from .build import build_dataset
from .base_dataset import Dataset
from .coco import coco
from .voc import voc