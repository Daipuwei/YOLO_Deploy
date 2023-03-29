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

    def __init__(self,logger,onnx_model_path,input_name,output_name,input_shape,output_shape):
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
        self.input_name = input_name
        self.output_name = output_name
        self.input_shape = input_shape
        self.output_shape = output_shape

        # 初始化ONNXRuntime推理引擎
        if os.path.exists(self.onnx_model_path):
            session_option = rt.SessionOptions()
            session_option.log_severity_level = 3
            self.session = rt.InferenceSession(self.onnx_model_path, sess_options=session_option,
                                                providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
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
            outputs = self.session.run([self.output_name], {self.input_name: input_tensor})[0]
            outputs = np.reshape(outputs,self.output_shape)
            return outputs
        except AttributeError as e:
            self.logger.debug(e)
            return None