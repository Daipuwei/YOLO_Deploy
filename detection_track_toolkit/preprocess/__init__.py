# -*- coding: utf-8 -*-
# @Time    : 2026/9/10 05:29
# @Author  : DaiPuWei
# @Email   : daipuwei@qq.com
# @File    : __init__.py
# @Software: PyCharm

"""
    这是预处理包的初始化脚本（与postprocess包同构：抽象基类→具体类→注册器→按名称构建）。

    预处理器工厂（yolov5/yolox）由 preprocess.build 导入风格模块时完成注册，
    上层通过 build_preprocessor 按名称创建预处理器实例。
"""

from .base_preprocessor import BasePreprocessor
from .build import PREPROCESSOR_REGISTRY
from .build import build_preprocessor
