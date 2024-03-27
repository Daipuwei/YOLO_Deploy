# -*- coding: utf-8 -*-
# @Time    : 2023/3/28 下午2:30
# @Author  : DaiPuWei
# @Email   : daipuwei@qq.com
# @File    : onnx_model.py
# @Software: PyCharm

"""
    这是定义ONNX推理引擎的脚本
"""

import os
import sys
import numpy as np
import onnxruntime as rt
from ..base_engine import BaseEngine

class ONNXEngine(BaseEngine):

    def __init__(self, logger, onnx_model_path,**kwargs):
        """
        这是ONNXRuntime推理引擎的初始化函数
        Args:
            logger： 日志类实例
            onnx_model_path: ONNX模型文件路径
        """
        assert os.path.exists(onnx_model_path), "ONNX模型不存在：{0}".format(onnx_model_path)
        super(ONNXEngine,self).__init__(logger,onnx_model_path)
        self.__dict__.update(kwargs)

        # 初始化模型参数
        self.input_names = []
        self.input_shapes = []
        self.input_types = []
        self.output_names = []
        self.output_shapes = []

        # 初始化ONNXRuntime推理引擎
        session_option = rt.SessionOptions()
        session_option.log_severity_level = 3
        # 初始化引擎
        self.session = rt.InferenceSession(self.engine_model_path, sess_options=session_option,
                                           #providers=['CUDAExecutionProvider', 'CPUExecutionProvider'],
                                           providers=['CUDAExecutionProvider'])
        # 初始化输入输出名称和形状
        for input_tensor in self.session.get_inputs():
            self.input_names.append(input_tensor.name)
            self.input_shapes.append(input_tensor.shape)
            self.input_types.append(input_tensor.type)

        # 输出名称和输出形状都没指定时才做读取输出节点信息
        for output_tensor in self.session.get_outputs():
            self.output_names.append(output_tensor.name)
            self.output_shapes.append(output_tensor.shape)
            self.output_types.append(output_tensor.type)

    def get_input_shape(self):
        """
        这是ONNX推理引擎获取模型输入形状的函数
        Returns:
        """
        return self.input_shapes

    def get_is_nchw(self):
        """
        这是ONNX推理引擎获取模型输入形状是否为nchw格式的函数
        Returns:
        """
        flag = True
        for input_shape in self.input_shapes:
            if len(input_shape) == 4:
                if input_shape[1] > 3:
                    flag = False
                break
        return flag

    # def __del__(self):
    #     del self.input_shape
    #     del self.input_name
    #     del self.output_shape
    #     del self.output_name
    #     del self.session

    def inference(self,input_tensor):
        """
        这是ONNX推理引擎的前向推理函数
        Args:
            input_tensor: 输入张量(列表)
        Returns:
        """
        # 初始化模型输入字典
        input_dict = {}
        for _input_tensor, _input_name, _input_type in zip(input_tensor, self.input_names, self.input_types):
            if _input_type == 'tensor(float)':
                input_dict[_input_name] = np.array(_input_tensor, dtype=np.float32)
            else:
                input_dict[_input_name] = np.array(_input_tensor, dtype=np.float16)
        # 模型推理获取输出结果
        outputs = self.session.run(self.output_names, input_dict)
        outputs = [np.reshape(_output, _output_shape) for _output, _output_shape in zip(outputs, self.output_shapes)]
        outputs = np.array(outputs, dtype=np.float32)
        return outputs
