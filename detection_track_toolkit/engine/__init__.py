# -*- coding: utf-8 -*-
# @Time    : 2026/9/8 11:
# 57
# @Author  : DaiPuWei
# @Email   : daipuwei@qq.com
# @File    : __init__.py.py
# @Software: PyCharm

from .base_engine import BaseEngine
from .build import ENGINE_REGISTRY, build_engine

# 注意：onnxruntime/tensorrt 引擎不再于此 star import（避免 import engine 时拉起
# onnxruntime/tensorrt 等重库），注册由 build_engine 按需惰性完成
