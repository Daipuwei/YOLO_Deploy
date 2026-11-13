# -*- coding: utf-8 -*-
# @Time    : 2026/9/10 06:33
# @Author  : DaiPuWei
# @Email   : daipuwei@qq.com
# @File    : yolox_preprocessor.py
# @Software: PyCharm

"""
    这是定义YOLOX风格预处理器的脚本。

    模块内容：
        1. YOLOXPreprocessor：YOLOX风格预处理类
           （keep_aspect_ratio等比例缩放→BGR直通→可选归一化→(转NCHW)），
           完整单图流水线由preprocess_single实现（基类抽象方法），
           yolox直接使用；
           其余变体经构造参数表达：
           - resize_mode='resize'：yolos直接拉伸缩放；
        2. yolox_preprocess(...)：预处理器工厂函数，注册进PREPROCESSOR_REGISTRY
           （注册名'yolox_preprocess'），由preprocess.build.build_preprocessor
           按名称创建。
"""

import numpy as np

from ..base_preprocessor import BasePreprocessor
from ..build import PREPROCESSOR_REGISTRY

class YOLOXPreprocessor(BasePreprocessor):
    """
    这是YOLOX风格预处理类（keep_aspect_ratio等比例缩放→BGR直通→可选归一化→(转NCHW)）
    """

    def __init__(self,input_shape,is_nchw=False,resize_mode="keep_aspect_ratio",
                 pad_mode="top_left"):
        """
        这是YOLOX风格预处理器的初始化函数
        Args:
            input_shape: 模型输入shape（4维，NCHW布局为[N,C,H,W]，
                         NHWC布局为[N,H,W,C]），由模型装配层从引擎读取后传入
            resize_mode: 缩放方式，'keep_aspect_ratio'或'resize'，默认为'keep_aspect_ratio'
                         （yolos传'resize'）
            is_nchw: 模型输入布局是否为NCHW（引擎判定结果由装配层原样传入），
                     默认为False
            pad_mode: letterbox填充锚定模式，'top_left'（缺省，填充全在右下）
                     或'center'（居中填充，ultralytics口径）；坐标还原偏移由
                     基类compute_restore_params按实际填充量给出
        """
        super(YOLOXPreprocessor, self).__init__(input_shape,is_nchw=is_nchw,
                                                resize_mode=resize_mode,
                                                pad_mode=pad_mode)

    def preprocess_single(self,image):
        """
        这是对单张图像做完整预处理的函数（YOLOX风格流水线：
        keep_aspect_ratio等比例缩放→BGR直通→可选归一化→(转NCHW)→连续内存化）
        Args:
            image: 图像，opencv格式
        Returns:
            image_tensor: 预处理后的图像张量
            image_shape: 原始图像尺度(height,width)
            restore_params: 缩放逆变换参数(gain_h,gain_w,pad_h,pad_w)
                （逐图w/h缩放系数与偏移系数，由基类compute_restore_params
                计算，交后处理坐标还原）
        """
        h,w = np.shape(image)[0:2]
        # 缩放
        image_tensor = self.resize_image(image)
        # 颜色转换：不做转换，BGR直通
        # 归一化：不做（仅转float32，数值原样透传）
        image_tensor = image_tensor.astype(np.float32)
        # hwc->chw
        if self.is_nchw:
            image_tensor = np.transpose(image_tensor, (2, 0, 1))
        image_tensor = np.ascontiguousarray(image_tensor)
        # 缩放逆变换参数（与resize_image实际变换互为逆变换，交后处理还原）
        restore_params = self.compute_restore_params(h, w)
        return image_tensor,(h,w),restore_params


@PREPROCESSOR_REGISTRY.register()
def yolox_preprocess(input_shape,is_nchw=False, **kwargs):
    """
    这是初始化YOLOX风格预处理器的工厂函数
    Args:
        input_shape: 模型输入shape（4维，NCHW布局为[N,C,H,W]，
                     NHWC布局为[N,H,W,C]），由模型装配层从引擎读取后传入
        is_nchw: 模型输入布局是否为NCHW（引擎判定结果由装配层原样传入），默认为False
        kwargs: 自定义参数字典（resize_mode，透传给预处理器类）
    Returns:
        YOLOXPreprocessor实例
    """
    return YOLOXPreprocessor(input_shape,is_nchw=is_nchw, **kwargs)
