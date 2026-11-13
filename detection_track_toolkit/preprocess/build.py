# -*- coding: utf-8 -*-
# @Time    : 2026/9/10 06:28
# @Author  : DaiPuWei
# @Email   : daipuwei@qq.com
# @File    : build.py
# @Software: PyCharm

"""
    这是定义预处理器注册器与按名称构建函数的脚本。

    模块内容：
        1. PREPROCESSOR_REGISTRY：预处理器注册器，各风格工厂函数
           （如yolov5_preprocess、yolox_preprocess）通过装饰器注册，
           注册名即工厂函数名；
        2. build_preprocessor(preprocessor_type, input_shape, **kwargs)：
           按名称创建预处理器，未知名称报ValueError并列出可用预处理器。
        风格模块（yolov5_preprocess/yolox_preprocess）在本模块末尾导入以触发注册
        （风格模块仅依赖cv2/numpy轻量库，与引擎重库不同，无需惰性导入）。
"""

from ..utils import Registry

PREPROCESSOR_REGISTRY = Registry(
    "PREPROCESSOR",
    doc="这是检测模型预处理器的注册器，用于按名称创建不同预处理风格，例如yolov5、yolox")


def build_preprocessor(preprocessor_type,input_shape,is_nchw=False,**kwargs):
    """
    这是按名称初始化检测模型预处理器的函数
    Args:
        preprocessor_type: 预处理器类型名称（注册名，统一转小写匹配，
           候选：yolov5_preprocess、yolox_preprocess）
        input_shape: 模型输入shape（4维，NCHW布局为[N,C,H,W]，
                     NHWC布局为[N,H,W,C]），由模型装配层从引擎读取后传入
        is_nchw: 模型输入布局是否为NCHW（引擎判定结果原样传入），默认为False
        kwargs: 自定义参数字典（透传给对应预处理器工厂）
    Returns:
        预处理器实例
    """
    preprocessor_type = str(preprocessor_type).lower()
    if preprocessor_type not in PREPROCESSOR_REGISTRY:
        raise ValueError("未知的预处理器类型 '{0}'，可用预处理器：{1}"
                         .format(preprocessor_type, sorted(PREPROCESSOR_REGISTRY.keys())))
    return PREPROCESSOR_REGISTRY.get(preprocessor_type)(input_shape,is_nchw=is_nchw,**kwargs)


# 导入风格模块触发工厂注册（置于末尾避免循环导入：风格模块反向导入
# PREPROCESSOR_REGISTRY；风格模块依赖轻量库，无需像引擎那样惰性导入）
from . import yolov5
from . import yolox
