# -*- coding: utf-8 -*-
# @Time    : 2024/4/8 下午10:10
# @Author  : DaiPuWei
# @Email   : daipuwei@qq.com
# @File    : build.py
# @Software: PyCharm

"""
    这是定义模型注册器的脚本
"""

from utils import Registry

MODEL_REGISTRY = Registry("MODEL")
MODEL_REGISTRY.__doc__ = """这是模型类的注册器，用于自动生成不同模型类，例如YOLOv5"""

def build_model(logger,cfg,**kwargs):
    """
    这是初始化模型的函数
    Args:
        logger: 日志类实例
        cfg: 参数配置字典
        kwargs: 自定义参数字典
    Returns:
    """
    model_type = cfg["model_type"]
    return MODEL_REGISTRY.get(model_type)(logger,cfg,**kwargs)
