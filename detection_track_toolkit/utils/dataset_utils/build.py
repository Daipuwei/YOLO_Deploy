# -*- coding: utf-8 -*-
# @Time    : 2026/9/10 11:02
# @Author  : DaiPuWei
# @Email   : daipuwei@qq.com
# @File    : build.py
# @Software: PyCharm

"""
    这是定义数据集注册器与按名称构建函数的脚本。

    模块内容：
        1. DATASET_REGISTRY：数据集注册器，各数据集工厂函数（如voc、coco、yolo）
           通过装饰器注册，注册名即工厂函数名（统一小写）；
        2. _LAZY_DATASET_MODULES：数据集目录下可惰性导入的模块白名单（动态发现，
           非数据集工具模块经_NON_DATASET_MODULES排除）；
        3. _DATASET_MODULE_MAP：数据集类型到模块的别名映射（同一模块注册
           多个数据集类型时使用，如yolo模块注册yolo_v1/yolo_v2两个类型）；
        4. build_dataset(dataset_type, dataset_dir, batchsize, mode, registry, **kwargs)：
           按名称构建数据集加载器：
           - dataset_type统一转小写匹配注册名，未知类型报ValueError并列出可用数据集；
           - 未注册时按白名单惰性导入对应模块触发注册（新数据集模块忘记在
             __init__.py接线时兜底）；
           - registry缺省使用模块级DATASET_REGISTRY，测试/隔离场景可注入；
           - kwargs透传给数据集工厂（如yolo的class_names_path）。
"""

import importlib
import os
import pkgutil

from .. import Registry

DATASET_REGISTRY = Registry(
    "DATASET",
    doc="这是数据集的注册器，用于自动生成不同数据集类，例如VOC、COCO、YOLO")

# 数据集目录下的可用模块白名单：build_dataset惰性注册时只允许导入这些模块，
# 既避免误导入工具模块，也防止配置注入任意模块路径
_DATASET_PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))
_NON_DATASET_MODULES = frozenset({"base_dataset", "build"})
_LAZY_DATASET_MODULES = {m.name for m in pkgutil.iter_modules([_DATASET_PACKAGE_DIR])
                         if not m.ispkg and m.name not in _NON_DATASET_MODULES}
# 数据集类型到模块的别名映射：同一模块注册多个数据集类型时使用
# （如yolo模块同时注册yolo_v1/yolo_v2两个类型），惰性导入按此映射查找模块
_DATASET_MODULE_MAP = {"yolo_v1": "yolo", "yolo_v2": "yolo"}


def build_dataset(dataset_type,dataset_dir,batchsize=1,mode='train',
                  registry=None,**kwargs):
    """
    这是搭建数据集加载器的函数
    Args:
        dataset_type: 数据集类型名称（注册名，统一转小写匹配）
        dataset_dir: 数据集地址
        batchsize: 小批量数据规模，默认为1
        mode: 子集类型，默认为'train'
        registry: 数据集注册器，缺省使用模块级DATASET_REGISTRY
                  （测试/隔离场景可注入自定义注册表）
        kwargs: 自定义参数字典（透传给对应数据集工厂，
                如yolo的class_names_path）
    Returns:
        数据集加载器实例
    """
    registry = DATASET_REGISTRY if registry is None else registry
    dataset_type = str(dataset_type).lower()
    # 未注册时按白名单惰性导入对应模块触发注册（注册断链兜底），
    # 仍无则报错列出可用数据集（配置错误语义用ValueError，KeyError偏"查找失败"）
    if dataset_type not in registry:
        module = _DATASET_MODULE_MAP.get(dataset_type, dataset_type)
        if module in _LAZY_DATASET_MODULES:
            importlib.import_module("." + module, __package__)
    if dataset_type not in registry:
        # 可用数据集列出已注册类型与可惰性导入的模块名；
        # 别名映射的源模块（如yolo）不是类型本身，从候选列表中剔除
        raise ValueError("未知的数据集类型 '{0}'，可用数据集：{1}"
                         .format(dataset_type,
                                 sorted(set(registry.keys()) |
                                        (_LAZY_DATASET_MODULES - set(_DATASET_MODULE_MAP.values())))))
    return registry.get(dataset_type)(dataset_dir,batchsize,mode,**kwargs)
