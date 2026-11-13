# -*- coding: utf-8 -*-
# @Time    : 2026/9/10 05:25
# @Author  : DaiPuWei
# @Email   : daipuwei@qq.com
# @File    : __init__.py
# @Software: PyCharm

"""
    这是后处理包的初始化脚本（与engine包同构：抽象基类→具体类→注册器→按名称构建）。

    后处理器工厂（yolov5/yolox/dflyolo）由本包导入风格模块时完成注册，
    上层通过 build_postprocessor 按名称创建后处理器实例。
    核心算法亦以独立函数导出（iou_xyxy/nms/rescale_coords），可脱离类直接调用。
"""

from .base_postprocessor import BasePostprocessor
from .anchor_base.yolov5_postprocess import YOLOv5Postprocessor
from .anchor_free.yolox_postprocess import YOLOXPostprocessor
from .anchor_free.dflyolo_postprocess import DFLYOLOPostprocessor
from .build import POSTPROCESSOR_REGISTRY
from .build import build_postprocessor
from .postprocess_utils import iou_xyxy
from .postprocess_utils import nms
from .postprocess_utils import rescale_coords
from .postprocess_utils import scale_coords
