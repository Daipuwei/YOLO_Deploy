# -*- coding: utf-8 -*-
# @Time    : 2026/9/10 05:26
# @Author  : DaiPuWei
# @Email   : daipuwei@qq.com
# @File    : build.py
# @Software: PyCharm

"""
    这是定义后处理器注册器与按名称构建函数的脚本。

    模块内容：
        1. POSTPROCESSOR_REGISTRY：后处理器注册器，各风格工厂函数
           （如yolov5_postprocess、yolox_postprocess、dflyolo_postprocess）
           通过装饰器注册，注册名即工厂函数名；
        2. build_postprocessor(postprocessor_type, ...)：按名称创建后处理器，
           未知名称报ValueError并列出可用后处理器。
        风格模块（yolov5_postprocess/yolox_postprocess/dflyolo_postprocess）
        在本模块末尾导入以触发注册（风格模块仅依赖cv2/numpy轻量库，
        与引擎重库不同，无需惰性导入）。
"""

from ..utils import Registry

POSTPROCESSOR_REGISTRY = Registry(
    "POSTPROCESSOR",
    doc="这是检测模型后处理器的注册器，用于按名称创建不同检测头后处理风格，"
        "例如yolov5、yolox、dflyolo")


def build_postprocessor(postprocessor_type,confidence_threshold,iou_threshold,
                        class_names,model_hw,**kwargs):
    """
    这是按名称初始化检测模型后处理器的函数
    Args:
        postprocessor_type: 后处理器类型名称（注册名，统一转小写匹配，
           候选：yolov5_postprocess、yolox_postprocess、dflyolo_postprocess）
        confidence_threshold: 置信度阈值
        iou_threshold: iou阈值
        class_names: 目标分类名称数组
        model_hw: 模型输入尺寸(height,width)
        kwargs: 自定义参数字典（透传给对应后处理器工厂）
    Returns:
        后处理器实例
    """
    postprocessor_type = str(postprocessor_type).lower()
    if postprocessor_type not in POSTPROCESSOR_REGISTRY:
        raise ValueError("未知的后处理器类型 '{0}'，可用后处理器：{1}"
                         .format(postprocessor_type, sorted(POSTPROCESSOR_REGISTRY.keys())))
    return POSTPROCESSOR_REGISTRY.get(postprocessor_type)(confidence_threshold,iou_threshold,
                                                          class_names,model_hw,**kwargs)


# 导入风格模块触发工厂注册（置于末尾避免循环导入：风格模块反向导入
# POSTPROCESSOR_REGISTRY；风格模块依赖轻量库，无需像引擎那样惰性导入）
from . import anchor_base
from . import anchor_free
