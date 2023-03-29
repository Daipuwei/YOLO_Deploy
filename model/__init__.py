# -*- coding: utf-8 -*-
# @Time    : 2023/3/28 下午2:29
# @Author  : DaiPuWei
# @Email   : daipuwei@qq.com
# @File    : __init__.py.py
# @Software: PyCharm


from .base_model import DetectionModel
from .onnx_model import ONNX_Engine
from .tensorrt_model import TensorRT_Engine
from .yolov5 import YOLOv5