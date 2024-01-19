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
            logger： 日志类实例
            onnx_model_path: ONNX模型文件路径
        """
        assert not os.path.exists(onnx_model_path), "ONNX模型不存在：{0}".format(onnx_model_path)
        # 初始化模型参数
        self.logger = logger
        self.onnx_model_path = onnx_model_path
        self.input_name = []
        self.input_shape = []
        self.input_type = []
        self.output_name = []
        self.output_shape = []

        # 初始化ONNXRuntime推理引擎
        session_option = rt.SessionOptions()
        session_option.log_severity_level = 3
        # 初始化引擎
        self.session = rt.InferenceSession(self.onnx_model_path, sess_options=session_option,
                                           providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
        # 初始化输入输出名称和形状
        for input_tensor in self.session.get_inputs():
            self.input_name.append(input_tensor.name)
            self.input_shape.append(input_tensor.shape)
            self.input_type.append(input_tensor.type)

        # 输出名称和输出形状都没指定时才做读取输出节点信息
        for output_tensor in self.session.get_outputs():
            self.output_name.append(output_tensor.name)
            self.output_shape.append(output_tensor.shape)

    def get_input_shape(self):
        return self.input_shape

    def __del__(self):
        del self.input_shape
        del self.input_name
        del self.output_shape
        del self.output_name
        del self.session

    def inference(self,input_tensor):
        """
        这是onnx模型推理的函数
        Args:
            input_tensor: 输入张量
        Returns:
        """
        # 初始化模型输入字典
        input_dict = {}
        for _input_tensor, _input_name, _input_type in zip(input_tensor, self.input_name, self.input_type):
            # input_dict[_input_name] = _input_tensor
            if _input_type == 'tensor(float)':
                input_dict[_input_name] = np.array(_input_tensor, dtype=np.float32)
            else:
                input_dict[_input_name] = np.array(_input_tensor, dtype=np.float16)
        # 模型推理获取输出结果
        outputs = self.session.run(self.output_name, input_dict)
        outputs = [np.reshape(_output, _output_shape) for _output, _output_shape in zip(outputs, self.output_shape)]
        # for i in np.arange(len(outputs)):
        #     outputs[i] = np.reshape(outputs[i],self.output_shape[i])
        outputs = np.array(outputs, dtype=np.float32)
        return outputs
