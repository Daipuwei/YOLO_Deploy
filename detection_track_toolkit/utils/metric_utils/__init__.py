# -*- coding: utf-8 -*-
# @Time    : 2026/9/11 08:09
# @Author  : DaiPuWei
# @Email   : daipuwei@qq.com
# @File    : __init__.py.py
# @Software: PyCharm


"""
    这是检测性能评估工具子包的初始化脚本：
    COCOeval口径函数位于coco_metric_utils.py，
    Ultralytics风格YOLO口径函数位于yolo_metric_utils.py。
    两种口径可并排输出对照（数值存在微小差异，属正常口径差异）。
"""

from .coco_metric_utils import summarize_eval
from .coco_metric_utils import eval_coco
from .coco_metric_utils import eval_per_class
from .coco_metric_utils import save_coco_json
from .yolo_metric_utils import compute_ap
from .yolo_metric_utils import match_predictions
from .yolo_metric_utils import ap_per_class
from .yolo_metric_utils import compute_yolo_metrics
from .yolo_metric_utils import ConfusionMatrix
from .yolo_metric_utils import format_confusion_matrix
