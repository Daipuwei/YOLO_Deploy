# -*- coding: utf-8 -*-
# @Time    : 2026/9/10 08:47
# @Author  : DaiPuWei
# @Email   : daipuwei@qq.com
# @File    : build.py
# @Software: PyCharm

"""
    这是定义模型注册器的脚本。

    模块内容：
        1. MODEL_REGISTRY：模型注册器，各模型工厂函数（如yolov5/yolov8/yolox）
           通过装饰器注册，注册名即工厂函数名；
        2. build_model(logger, cfg, **kwargs)：模型构建入口：
           - cfg 必须含 model_type 键，取值统一转小写匹配注册名；
           - 未知模型报ValueError并列出可用模型；
           - kwargs 透传给模型工厂（gpu_id由模型类读取，其余键注入模型属性，
             冲突时DetectionModel抛ValueError）。
        模型模块（anchor_base/anchor_free）在本模块末尾导入以触发注册
        （模型模块仅依赖cv2/numpy轻量库，重库仍由引擎层惰性导入）。
"""

from ..utils import Registry

MODEL_REGISTRY = Registry("MODEL")
MODEL_REGISTRY.__doc__ = """这是模型类的注册器，用于自动生成不同模型类，例如YOLOv5"""


def build_model(logger,cfg,**kwargs):
    """
    这是初始化模型的函数
    Args:
        logger: 日志类实例
        cfg: 参数配置字典（model_type必填）
        kwargs: 自定义参数字典（gpu_id等，透传给模型工厂）
    Returns:
        检测模型类实例
    """
    model_type = cfg.get("model_type")
    if model_type is None:
        raise ValueError("配置中缺少model_type键，可用模型：{0}"
                         .format(sorted(MODEL_REGISTRY.keys())))
    model_type = str(model_type).lower()
    if model_type not in MODEL_REGISTRY:
        raise ValueError("未知的模型类型 '{0}'，可用模型：{1}"
                         .format(model_type, sorted(MODEL_REGISTRY.keys())))
    return MODEL_REGISTRY.get(model_type)(logger,cfg,**kwargs)


# 导入模型模块触发工厂注册（置于末尾避免循环导入：模型模块反向导入
# MODEL_REGISTRY/DetectionModel与engine/preprocess/postprocess包；
# 模型模块依赖轻量库，重库仍由引擎层惰性导入）
from . import anchor_base
from . import anchor_free
