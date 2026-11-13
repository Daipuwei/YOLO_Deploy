# -*- coding: utf-8 -*-
# @Time    : 2026/9/10 08:43
# @Author  : DaiPuWei
# @Email   : daipuwei@qq.com
# @File    : __init__.py.py
# @Software: PyCharm


"""
    这是DFL家族子包的初始化脚本（yolov6/v8/v9/v10/v11/v12/v13/v26/ppyoloe
    统一由DFLYOLO类装配，各注册工厂定义于dflyolo.py）
"""

from .dflyolo import DFLYOLO
from .dflyolo import yolov6,yolov8,yolov9,yolov10,yolov11,yolov12,yolov13,yolo26,ppyoloe
from .yolox import YOLOX
from .yolox import yolox