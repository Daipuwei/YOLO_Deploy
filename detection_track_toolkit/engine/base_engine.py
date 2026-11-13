# -*- coding: utf-8 -*-
# @Time    : 2026/9/9 06:53
# @Author  : DaiPuWei
# @Email   : daipuwei@qq.com
# @File    : base_engine.py
# @Software: PyCharm

"""
    这是定义抽象推理引擎的脚本。

    模块内容：
        1. BaseEngine：抽象推理引擎基类（ABC），定义所有推理引擎的统一契约：
           - 公共属性：logger、engine_model_path 与输入输出元信息列表
             （input_names/input_shapes/input_types/output_names/output_shapes/
             output_types），由 __init__ 统一初始化，子类直接复用；
           - 模型文件路径统一解析（绝对原样；相对路径CWD优先、仓库根兜底，
             见utils.common_utils.resolve_config_path）与存在性检查在基类统一
             完成（FileNotFoundError，不走assert，python -O 下行为一致）；
           - kwargs 逐键注入为引擎属性（下划线开头或覆盖已有属性的键抛ValueError）；
           - inference 为抽象方法，子类必须实现，否则实例化时抛TypeError
             （而不是运行时静默返回None）；
           - get_input_shape/get_is_nchw 提供基类默认实现：
             get_input_shape 返回输入形状列表的浅拷贝（调用方修改不影响引擎内部状态），
             get_is_nchw 由 judge_is_nchw 统一判定；
           - judge_is_nchw 静态方法：所有引擎共用的输入布局（NCHW/NHWC）统一启发式；
           - release()：资源释放契约，缺省空实现，持有GPU上下文等资源的子类应覆写。

        子类：engine.onnxruntime.onnxruntime_engine.ONNXRuntimeEngine、
              engine.tensorrt.tensorrt_engine.TensorRTEngine，
        均由 engine.build.build_engine 按需惰性导入并注册到 ENGINE_REGISTRY。
"""

import os
import numpy as np
from typing import Any, List
from abc import ABC, abstractmethod

from ..utils import resolve_config_path


def format_io_node_info(direction,name,shape,dtype):
    """
    这是格式化输入输出节点信息的函数（初始化完成后逐个节点按行打印，
    适配多输入多输出模型）
    Args:
        direction: 节点方向，'输入节点'或'输出节点'
        name: 节点名称
        shape: 节点形状列表
        dtype: 节点dtype（numpy dtype或其name字符串）
    Returns:
        格式化后的节点信息字符串，形如
        '输入节点images: dim=1x480x480x3, dtype=float32, dtype_size=4,
         dim_size=2764800, size=11059200'（动态维时dim_size与size为-1）
    """
    dim_str = "x".join(str(d) for d in shape)
    np_dtype = np.dtype(dtype)
    if all(isinstance(d, int) and d > 0 for d in shape):
        dim_size = int(np.prod(shape))
        size = dim_size * np_dtype.itemsize
    else:
        # 动态维：元素数与字节数无法计算，置-1
        dim_size, size = -1, -1
    return "{0}{1}: dim={2}, dtype={3}, dtype_size={4}, dim_size={5}, size={6}".format(
        direction, name, dim_str, np_dtype.name, np_dtype.itemsize, dim_size, size)


class BaseEngine(ABC):

    def __init__(self, logger: Any, engine_model_path: str, **kwargs: Any):
        """
        这是抽象推理引擎的初始化函数
        Args:
            logger: 日志类实例
            engine_model_path: 推理引擎模型文件路径
            kwargs: 自定义参数字典（逐键注入为引擎属性；键不能以下划线开头，
                    也不能与已有属性同名，否则抛ValueError）
        """
        # 模型文件路径统一解析（绝对原样；相对路径CWD优先、仓库根兜底，
        # 见resolve_config_path）后做存在性校验：修复配置内相对路径
        # （./model_data/xxx.onnx）曾按进程CWD解析，非仓库根目录运行时
        # 直接FileNotFoundError；解析结果存入实例属性，两个引擎子类
        # 均按self.engine_model_path打开模型文件，无需各自重复解析
        engine_model_path = resolve_config_path(engine_model_path)
        # 模型文件检查：不用assert（python -O下断言会被剥离），所有引擎统一在此校验
        if not os.path.exists(engine_model_path):
            raise FileNotFoundError("模型文件不存在：{0}".format(engine_model_path))
        # 初始化相关变量
        self.logger = logger
        self.engine_model_path = engine_model_path

        # 初始化模型参数
        self.input_names = []
        self.input_shapes = []
        self.input_types = []
        self.output_names = []
        self.output_shapes = []
        self.output_types = []

        # 其余kwargs注入为属性：逐键检查，拒绝覆盖已有属性（拼错键/冲突键立即报错，
        # 而不是静默接受后在使用处才炸）
        for key, value in kwargs.items():
            if key.startswith("_") or hasattr(self, key):
                raise ValueError("非法或冲突的引擎配置键'{0}'：不能以下划线开头或覆盖已有属性"
                                 .format(key))
            setattr(self, key, value)

    @staticmethod
    def judge_is_nchw(input_shapes: List[Any]) -> bool:
        """
        这是统一判定模型输入布局是否为NCHW的静态函数（所有引擎共用同一启发式，
        避免各子类各写一套且判定方向不一致）
        判定规则：对4维输入，第4维为具体正整数且<=3视为NHWC（通道在末维）；
        否则第2维为具体正整数且<=3视为NCHW；均无法判断时缺省NCHW。
        Args:
            input_shapes: 模型输入形状列表
        Returns:
            是否NCHW布局（bool）
        """
        for input_shape in input_shapes:
            if len(input_shape) != 4:
                continue
            dim1, dim3 = input_shape[1], input_shape[3]
            if isinstance(dim3, int) and 0 < dim3 <= 3:
                return False
            if isinstance(dim1, int) and 0 < dim1 <= 3:
                return True
        return True

    def get_input_shape(self) -> List[Any]:
        """
        这是抽象推理引擎获取模型输入形状的函数
        Returns:
            模型输入形状列表（浅拷贝：调用方修改返回列表不影响引擎内部状态）
        """
        return list(self.input_shapes)

    def get_is_nchw(self) -> bool:
        """
        这是抽象推理引擎获取模型输入布局是否为NCHW的函数
        Returns:
            是否NCHW布局（bool，由judge_is_nchw统一判定）
        """
        return self.judge_is_nchw(self.input_shapes)

    @abstractmethod
    def inference(self, input_tensor: List[Any]) -> List[Any]:
        """
        这是抽象推理引擎前向推理函数（抽象方法，子类必须实现；
        基类无法实例化，漏实现会在实例化时报TypeError而不是运行时返回None）
        Args:
            input_tensor: 输入张量列表（每元素对应模型一路输入）
        Returns:
            输出张量列表（每路输出一个数组，顺序与模型输出一致）
        """
        raise NotImplementedError

    def release(self):
        """
        这是释放引擎资源的函数（契约声明，缺省空实现；持有GPU上下文等资源的
        子类应覆写本方法，上层在替换/销毁引擎时统一调用）
        """

