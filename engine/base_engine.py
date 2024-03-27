# -*- coding: utf-8 -*-
# @Time    : 2024/3/20 下午3:43
# @Author  : DaiPuWei
# @Email   : daipuwei@qq.com
# @File    : base_engine.py
# @Software: PyCharm

"""
    这是定义抽象推理引擎的脚本
"""

class BaseEngine(object):

    def __init__(self, logger, engine_model_path,**kwargs):
        """
        这是抽象推理引擎的初始化函数
        Args:
            logger： 日志类实例
            engine_model_path: 推理引擎模型文件路径
            kwargs: 参数字典
        """
        # 初始化相关变量
        self.logger = logger
        self.engine_model_path = engine_model_path
        self.__dict__.update(kwargs)

        # 初始化模型参数
        self.input_names = []
        self.input_shapes = []
        self.input_types = []
        self.output_names = []
        self.output_shapes = []
        self.output_types = []

    def get_input_shape(self):
        """
        这是抽象推理引擎获取模型输入形状的函数
        Returns:
        """
        pass

    def get_is_nchw(self):
        """
        这是抽象推理引擎获取输入形状是否为nchw格式的函数
        Returns:
        """

    def inference(self,input_tensor):
        """
        这是抽象推理引擎前向推理函数
        Args:
            input_tensor: 输入张量
        Returns:
        """
        pass
