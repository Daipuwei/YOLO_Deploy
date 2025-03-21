# -*- coding: utf-8 -*-
# @Time    : 2024/4/8 下午9:47
# @Author  : DaiPuWei
# @Email   : daipuwei@qq.com
# @File    : build.py
# @Software: PyCharm

"""
    这是定义推理引擎类注册创建函数的脚本
"""

from utils import Registry

ENGINE_REGISTRY = Registry("ENGINE")
ENGINE_REGISTRY.__doc__ = """这是推理引擎类的注册器，用于自动生成不同推理引擎类，例如ONNXEngine、TensorRTEngine"""

def build_engine(logger,cfg,**kwargs):
    """
    这是初始化推理引擎的函数
    Args:
        logger: 日志类实例
        cfg: 参数配置字典
        **kwargs: 自定义参数字典
    Returns:
    """
    engine_type = cfg["engine_type"]
    return ENGINE_REGISTRY.get(engine_type)(logger,cfg,**kwargs)
