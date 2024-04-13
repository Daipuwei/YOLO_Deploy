# -*- coding: utf-8 -*-
# @Time    : 2023/3/28 下午2:29
# @Author  : DaiPuWei
# @Email   : daipuwei@qq.com
# @File    : __init__.py.py
# @Software: PyCharm

from .base_model import DetectionModel
from .build import MODEL_REGISTRY,build_model
from .yolov5.yolov5 import *
from .yolov8.yolov8 import *
from .yolos.yolos import *

