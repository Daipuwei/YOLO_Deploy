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

class ONNX_Engine(object):

    def __init__(self, logger, onnx_model_path):
        """
        这是ONNXRuntime推理引擎的初始化函数
        Args:
            onnx_model_path: ONNX模型文件路径
            input_name: 输入节点名称
            output_name: 输出节点名称
            input_shape: 输入节点shape
            output_shape: 输出节点shape
        """
        # 初始化模型参数
        self.logger = logger
        self.onnx_model_path = onnx_model_path
        self.input_name = []
        self.output_name = []
        self.input_shape = []
        self.output_shape = []

        # 初始化ONNXRuntime推理引擎
        if os.path.exists(self.onnx_model_path):
            session_option = rt.SessionOptions()
            session_option.log_severity_level = 3
            # 初始化引擎
            self.session = rt.InferenceSession(self.onnx_model_path, sess_options=session_option,
                                                providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
            # 初始化输入输出名称和形状
            input_tensors = self.session.get_inputs()
            for input_tensor in input_tensors:
                self.input_name.append(input_tensor.name)
                self.input_shape.append(input_tensor.shape)
            output_tensors = self.session.get_outputs()
            for output_tensor in output_tensors:
                self.output_name.append(output_tensor.name)
                self.output_shape.append(output_tensor.shape)
        else:
            self.logger.info("ONNX文件不存在：{0}".format(self.onnx_model_path))
            sys.exit()

    def inference(self,input_tensor):
        """
        这是onnx模型推理的函数
        Args:
            input_tensor: 输入张量
        Returns:
        """
        # 模型推理
        try:
            input_dict = {}
            for _input_tensor,_input_name in zip(input_tensor,self.input_name):
                input_dict[_input_name] = _input_tensor
            outputs = self.session.run(self.output_name, input_dict)
            for i in np.arange(len(outputs)):
                outputs[i] = np.reshape(outputs[i],self.output_shape[i])
            # outputs = self.session.run([self.output_name], {self.input_name: input_tensor})[0]
            # outputs = np.reshape(outputs,self.output_shape)
            return outputs
        except AttributeError as e:
            self.logger.debug(e)
            return None
