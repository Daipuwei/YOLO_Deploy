# -*- coding: utf-8 -*-
# @Time    : 2023/3/28 下午2:29
# @Author  : DaiPuWei
# @Email   : daipuwei@qq.com
# @File    : __init__.py.py
# @Software: PyCharm

from .base_model import DetectionModel
from .build import MODEL_REGISTRY,build_model
from .yolov4 import *
from .yolov5 import *
from .yolov8 import *
from .yolov9 import *
from .yolov10 import *
from .yolov11 import *
from .yolov12 import *
from .yolos import *
from .yolox import *
