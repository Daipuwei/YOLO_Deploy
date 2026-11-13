# -*- coding: utf-8 -*-
# @Time    : 2026/9/9 01:13
# @Author  : DaiPuWei
# @Email   : daipuwei@qq.com
# @File    : __init__.py
# @Software: PyCharm

"""
    这是检测模型包的初始化脚本（抽象基类→具体类→注册器→按名称构建）。

    模型工厂（yolov5/yolov7/yolox/yolov6/v8/v9/v10/v11/v12/v13/v26/ppyoloe）
    由 detection_model.build 末尾导入 anchor_base/anchor_free 时完成注册，
    上层通过 build_model 按名称装配模型。
    get_classes/draw_detection_results/random_generate_colors 等共享检测工具
    亦在此导出（原 utils.detection_utils 已迁移至本包 detection_utils.py）。
"""

from .base_model import DetectionModel
from .build import MODEL_REGISTRY
from .build import build_model
from .detection_utils import get_classes
from .detection_utils import random_generate_colors
from .detection_utils import draw_detection_results
from .detection_utils import x1y1x2y2_to_x1y1wh
