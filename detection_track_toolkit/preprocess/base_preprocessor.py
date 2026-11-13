# -*- coding: utf-8 -*-
# @Time    : 2026/9/10 06:28
# @Author  : DaiPuWei
# @Email   : daipuwei@qq.com
# @File    : base_preprocessor.py
# @Software: PyCharm

"""
    这是定义检测模型图像预处理器抽象基类的脚本。

    模块内容：
        1. BasePreprocessor：预处理器抽象基类（ABC）：
           - 输入shape与布局由构造参数传入（与推理引擎解耦，由模型装配层
             从引擎读取后传入；布局即引擎根据shape判定的is_nchw结果，
             预处理块不再重复判定），构造时校验并解包，
             通过batch_size/c/h/w/is_nchw属性对外暴露；
           - resize_mode：缩放方式，'keep_aspect_ratio'等比例缩放填充（缺省）
             或'resize'直接拉伸填满；
           - pad_mode：letterbox填充锚定模式，'top_left'缩放图贴左上角、
             填充全在右下（缺省，坐标还原无偏移）或'center'居中填充
             （ultralytics口径，坐标还原需减填充偏移）；仅
             resize_mode='keep_aspect_ratio'时生效；
           - is_nchw：模型输入布局是否为NCHW，据此决定是否hwc2chw
             （yolov4/yolov8等不转NCHW的特殊导出模型由装配层传False）；
           - resize_image(image)：缩放共享辅助函数（keep_aspect_ratio/resize，
             两种风格预处理共用，letterbox按pad_mode锚定填充）；
           - compute_restore_params(h, w)：计算单图缩放逆变换参数
             （restore_params=(gain_h,gain_w,pad_h,pad_w)，即逐图w/h的
             缩放系数与偏移系数；统一逆变换orig=(model-pad)/gain按轴独立。
             letterbox取实际整像素缩放与填充量（两种pad_mode均支持），
             resize拉伸双轴独立gain无填充——后处理侧rescale_coords据此
             精确还原，不再按min gain反推，拉伸模式与居中填充由此可
             正确还原）；
           - preprocess_single(image)为抽象方法，子类必须实现
             （完整单图预处理流水线的差异所在：缩放→颜色转换→归一化→
             按布局转NCHW），返回(图像张量,(原始h,w),restore_params)；
           - preprocess(batch_images)为总体预处理入口，逐张调用
             preprocess_single并堆叠成批量张量，restore_params逐图收集成
             (batchsize,4)数组，与原始图像尺度一并交后处理器做检测框
             尺度还原（三块契约：preprocess返回
             (image_tensor,image_shapes,restore_params)三元组）。

        子类：preprocess.yolov5.yolov5_preprocessor.YOLOv5Preprocessor（YOLOv5风格）、
              preprocess.yolox.yolox_preprocessor.YOLOXPreprocessor（YOLOX风格），
        均由 preprocess.build.PREPROCESSOR_REGISTRY 注册工厂函数，
        由 build_preprocessor 按名称创建。
"""

import cv2
import numpy as np

from abc import ABC, abstractmethod

from .preprocess_utils import letterbox_params
from .preprocess_utils import resize_keep_aspect_ratio
from .preprocess_utils import resize

class BasePreprocessor(ABC):

    def __init__(self,input_shape,is_nchw=False,resize_mode="keep_aspect_ratio",
                 pad_mode="top_left"):
        """
        这是检测模型预处理器抽象基类的初始化函数
        Args:
            input_shape: 模型输入shape（4维，NCHW布局为[N,C,H,W]，
                         NHWC布局为[N,H,W,C]），由模型装配层从引擎读取后传入
            is_nchw: 模型输入布局是否为NCHW（引擎已根据input_shape判定，
                     由模型装配层原样传入；yolov4/yolov8等不转NCHW的特殊
                     导出模型传False）
            resize_mode: 缩放方式，'keep_aspect_ratio'等比例缩放填充
                         （缺省）或'resize'直接拉伸填满目标尺寸
            pad_mode: letterbox填充锚定模式，'top_left'（缺省，缩放图贴
                      左上角、填充全在右下，坐标还原无偏移）或'center'
                      （居中填充，ultralytics口径，坐标还原需减填充偏移）；
                      仅resize_mode='keep_aspect_ratio'时生效
        """
        self.input_shape = input_shape
        if resize_mode not in ("keep_aspect_ratio", "resize"):
            raise ValueError("未知的缩放模式'{0}'，支持keep_aspect_ratio或resize"
                             .format(resize_mode))
        if pad_mode not in ("top_left", "center"):
            raise ValueError("未知的填充锚定模式'{0}'，支持top_left或center"
                             .format(pad_mode))
        self.resize_mode = resize_mode
        self.pad_mode = pad_mode
        self.is_nchw = is_nchw
        # 校验模型输入shape：仅支持4维且各维为正整数，动态维度提前明确报错
        if len(input_shape) != 4:
            raise ValueError("模型输入shape必须为4维（NCHW或NHWC），实际为{0}"
                             .format(input_shape))
        if self.is_nchw:
            self.batch_size, self.c, self.h, self.w = input_shape
        else:
            self.batch_size, self.h, self.w, self.c = input_shape
        for dim in (self.batch_size, self.c, self.h, self.w):
            if not isinstance(dim, int) or dim <= 0:
                raise ValueError("模型输入shape各维必须为正整数，实际为{0}"
                                 .format(input_shape))

    def resize_image(self,image):
        """
        这是将图像缩放到模型输入尺寸的共享辅助函数（两种风格预处理共用）
        Args:
            image: 图像，opencv格式
        Returns:
            缩放后的图像（图像尺寸与模型输入尺寸一致时直接返回原图，
            零拷贝跳过缩放/填充）
        """
        h, w = np.shape(image)[0:2]
        # 尺寸一致直接返回原图：跳过缩放与填充（视频检测时输入分辨率
        # 恰为模型输入尺寸的场景收益明显）
        if h == self.h and w == self.w:
            return image
        if self.resize_mode == "keep_aspect_ratio":
            # 等比例缩放并填充像素（按pad_mode锚定填充位置）
            return resize_keep_aspect_ratio(image, (self.h, self.w),
                                            pad_mode=self.pad_mode)
        else:
            # 直接拉伸填满模型输入尺寸
            return resize(image, (self.h, self.w))

    def compute_restore_params(self,h,w):
        """
        这是计算单图缩放逆变换参数的函数（与resize_image的实际变换互为
        精确逆变换：填充执行与坐标还原共用letterbox_params同一组参数，
        后处理侧rescale_coords按orig=(model-pad)/gain逐轴还原）
        Args:
            h: 原始图像高度
            w: 原始图像宽度
        Returns:
            restore_params四元组(gain_h,gain_w,pad_h,pad_w)：
            - keep_aspect_ratio（letterbox）：取实际整像素缩放
              (gain_h=new_h/h, gain_w=new_w/w)与填充起点
              (pad_h=top, pad_w=left；top_left模式恒为0，center模式
              为居中填充偏移)；
            - resize（拉伸）：双轴独立gain(H/h, W/w)，无填充pad恒为0
        """
        if self.resize_mode == "keep_aspect_ratio":
            new_h, new_w, top, _, left, _ = letterbox_params(
                (h, w), (self.h, self.w), self.pad_mode)
            return (new_h / h, new_w / w, float(top), float(left))
        return (self.h / h, self.w / w, 0.0, 0.0)

    def release(self):
        """
        这是释放预处理器资源的函数（契约声明，缺省空实现：
        当前预处理器仅持有纯CPU配置，无原生资源；未来持有
        GPU缓冲/文件句柄等资源的子类应覆写本方法，
        模型侧release()会统一级联调用）
        """

    @abstractmethod
    def preprocess_single(self,image):
        """
        这是对单张图像做完整预处理的抽象函数（子类必须实现，风格差异所在：
        缩放→颜色转换→归一化→(转NCHW)→连续内存化的完整流水线由子类编排）
        Args:
            image: 图像，opencv格式
        Returns:
            image_tensor: 预处理后的图像张量
            image_shape: 原始图像尺度(height,width)
            restore_params: 缩放逆变换参数(gain_h,gain_w,pad_h,pad_w)
                （由compute_restore_params计算，交后处理坐标还原）
        """
        raise NotImplementedError

    def preprocess(self,batch_images):
        """
        这是对批量图像做完整预处理的函数（总体预处理入口）
        Args:
            batch_images: 批量图像数组，每张图像opencv读入
        Returns:
            image_tensor: 预处理后的批量图像张量
            image_shapes: 图像尺度数组，shape为(batchsize,2)
            restore_params: 逐图缩放逆变换参数数组，shape为(batchsize,4)
                （每行为(gain_h,gain_w,pad_h,pad_w)——逐图w/h的缩放系数与
                偏移系数，后处理坐标还原用；三块契约：本三元组经
                DetectionModel.detect透传给
                postprocess(outputs,image_shapes,restore_params)）
        """
        image_tensor = []
        image_shapes = []
        restore_params = []
        for image in batch_images:
            # 单独处理一张图像（完整流水线由子类实现）
            _image_tensor,_image_shape,_restore_params = self.preprocess_single(image)
            image_tensor.append(_image_tensor)
            image_shapes.append(_image_shape)
            restore_params.append(_restore_params)
        image_tensor = np.ascontiguousarray(image_tensor)
        image_shapes = np.array(image_shapes)
        restore_params = np.array(restore_params, dtype=np.float32)
        return image_tensor,image_shapes,restore_params
