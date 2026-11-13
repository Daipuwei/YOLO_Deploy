# -*- coding: utf-8 -*-
# @Time    : 2026/9/8 11:47
# @Author  : DaiPuWei
# @Email   : daipuwei@qq.com
# @File    : __init__.py
# @Software: PyCharm

"""
    这是utils包的初始化脚本
    采用惰性导出（PEP 562）：包初始化时不再 star import 全部子模块，
    from utils import xxx 首次访问时才加载对应子模块，避免仅使用轻量工具
    （如 Registry）时连带拉起 cv2 等重依赖。

    注意：
        1.符号解析按"轻量优先"顺序探测子模块；探测到的子模块缺失时跳过继续
          （模块迁移后 _SUBMODULES 残留旧名时不再毒化全部导出）；
        2.不再支持 from utils import *（仓库无此用法）；
        3.新增符号无需维护映射表，名字唯一即可自动解析；同名时先探测到的
          子模块生效。

    迁移记录：detection_utils 已迁移至 detection_model/detection_utils.py，
    本包不再探测（get_classes/draw_detection_results 等符号请从
    detection_track_toolkit.detection_model.detection_utils 导入）。
"""

import importlib

# 子模块搜索顺序：轻量优先，拉 cv2 的 dataset_utils 垫底
_SUBMODULES = ("registry_utils", "json_utils", "common_utils",
               "logger_utils", "dataset_utils", "metric_utils")


def __getattr__(name):
    """惰性导出：from utils import xxx 时按序探测子模块（与原始 import * 语义一致）。"""
    if name.startswith("_"):
        raise AttributeError("module 'utils' has no attribute '{0}'".format(name))
    for sub in _SUBMODULES:
        try:
            module = importlib.import_module("." + sub, __name__)
        except ModuleNotFoundError as e:
            # 仅当缺失的是子模块本身（已迁移/删除）才跳过继续探测；
            # 子模块存在但其第三方依赖缺失（如metric_utils缺pycocotools、
            # logger_utils缺loguru）时原样抛出——修复：曾被无条件吞掉，
            # 最终报出误导性AttributeError把排障方向带偏
            if e.name != "{0}.{1}".format(__name__, sub):
                raise
            continue
        if hasattr(module, name):
            return getattr(module, name)
    raise AttributeError(
        "module 'utils' has no attribute '{0}'（请确认该符号所在子模块已加入 "
        "utils/__init__.py 的 _SUBMODULES）".format(name))
