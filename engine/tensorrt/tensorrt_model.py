# -*- coding: utf-8 -*-
# @Time    : 2023/3/28 下午2:31
# @Author  : DaiPuWei
# @Email   : daipuwei@qq.com
# @File    : tensorrt_model.py
# @Software: PyCharm

"""
    这是定义TensorRT推理引擎的脚本
"""

import os
import sys
import numpy as np
import pycuda.autoinit
import pycuda.driver as cuda
import tensorrt as trt
from ..base_engine import BaseEngine

class HostDeviceMem(object):
    """ Host and Device Memory Package """

    def __init__(self, host_mem, device_mem):
        self.host = host_mem
        self.device = device_mem

    def __str__(self):
        return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)

    def __repr__(self):
        return self.__str__()

class TensorRTEngine(BaseEngine):

    def __init__(self, logger, tensorrt_model_path,gpu_idx=0,**kwargs):
        """
        这是TensorRT模型推理引擎的初始化函数
        Args:
            logger: 日志类实例
            tensorrt_model_path: TensorRT模型文件路径
            gpu_idx: gpu序号,默认为0
        """
        assert os.path.exists(tensorrt_model_path), "TensorRT模型不存在：{0}".format(tensorrt_model_path)
        super(TensorRTEngine,self).__init__(logger,tensorrt_model_path)
        self.__dict__.update(kwargs)
        # 初始化模型参数
        self.input_shape = []
        self.output_shape = []

        # 初始化TensorRT推理引擎相关变量
        self.device_ctx = cuda.Device(gpu_idx).make_context()
        self.engine = self.load_tensorrt_engine()
        self.context = self.engine.create_execution_context()
        self.input, self.output, self.bindings, self.stream = self.allocate_buffers(self.context)

    def get_input_shape(self):
        """
        这是TensorRT推理引擎获取模型输入形状的函数
        Returns:
        """
        return self.input_shape

    def get_is_nchw(self):
        """
        这是TensorRT推理引擎获取模型输入形状是否为nchw格式的函数
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
    #     #self.device_ctx.detach()  # 2. 实例释放时需要detech cuda上下文
    #     del self.engine
    #     del self.context
    #     del self.bindings
    #     del self.stream
    #     del self.input
    #     del self.output

    # def __del__(self):
    #     del self.input
    #     del self.output
    #     del self.stream
    #     self.device_ctx.detach()  # release device context

    def load_tensorrt_engine(self):
        """
        这是加载TensorRT模型的函数
        Returns:
        """
        TRT_LOGGER = trt.Logger()
        with open(self.engine_model_path, "rb") as f, \
                trt.Runtime(TRT_LOGGER) as runtime:
            engine = runtime.deserialize_cuda_engine(f.read())
        self.logger.info("Loaded TensorRT engine from file {}".format(self.engine_model_path))
        return engine

    def allocate_buffers(self, context):
        """
        这是为TensorRT推理引擎分配模型输入输出缓存的函数
        Args:
            context: 上下文信息
        Returns:
        """
        inputs = []
        outputs = []
        bindings = []
        stream = cuda.Stream()
        for binding in self.engine:
            size = trt.volume(self.engine.get_binding_shape(binding))
            dtype = trt.nptype(self.engine.get_binding_dtype(binding))
            # Allocate host and device buffers
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            # Append the device buffer to device bindings.
            bindings.append(int(device_mem))
            # Append to the appropriate list.
            if self.engine.binding_is_input(binding):
                inputs.append(HostDeviceMem(host_mem, device_mem))
                binding_idx = self.engine.get_binding_index(binding)
                self.input_shape.append(self.engine.get_binding_shape(binding_idx))
            else:
                outputs.append(HostDeviceMem(host_mem, device_mem))
                binding_idx = self.engine.get_binding_index(binding)
                self.output_shape.append(self.engine.get_binding_shape(binding_idx))
        return inputs, outputs, bindings, stream

    def inference(self,input_tensor):
        """
        这是TensorRT推理引擎的前向推理函数
        Args:
            input_tensor: 输入张量列表
        Returns:
        """
        # Push to device
        self.device_ctx.push()
        # Copy data to input memory buffer
        [np.copyto(_inp.host, _input_tensor.ravel()) for _inp,_input_tensor in zip(self.input,input_tensor)]
        # Transfer input data to the GPU.
        # cuda.memcpy_htod_async(self._input.device, self._input.host, self._stream)
        [cuda.memcpy_htod_async(inp.device, inp.host, self.stream) for inp in self.input]
        # Run inference.
        self.context.execute_async_v2(bindings=self.bindings, stream_handle=self.stream.handle)
        # Transfer predictions back from the GPU.
        # cuda.memcpy_dtoh_async(self._output.host, self._output.device, self._stream)
        [cuda.memcpy_dtoh_async(out.host, out.device, self.stream) for out in self.output]
        # Synchronize the stream
        self.stream.synchronize()
        host_outputs = [out.host.reshape(output_shape) for out,output_shape in zip(self.output,self.output_shape)]
        # Pop the device
        self.device_ctx.pop()
        return host_outputs
