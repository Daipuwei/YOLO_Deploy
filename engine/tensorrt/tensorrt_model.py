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
#import pycuda.autoinit
import pycuda.driver as cuda
import tensorrt as trt
import pycuda.autoinit

class HostDeviceMem(object):
    """ Host and Device Memory Package """

    def __init__(self, host_mem, device_mem):
        self.host = host_mem
        self.device = device_mem

    def __str__(self):
        return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)

    def __repr__(self):
        return self.__str__()

class TensorRT_Engine(object):

    def __init__(self, logger, onnx_model_path, tensorrt_model_path,gpu_idx=0, mode="fp32", trt_int8_calibrator=None):
        """
        这是TensorRT模型推理引擎的初始化函数
        Args:
            onnx_model_path: ONNX模型文件路径
            tensorrt_model_path: TensorRT模型文件路径
            input_name: 输入节点名称
            output_name: 输出节点名称
            input_shape: 输入节点shape
            output_shape: 输出节点shape
            gpu_idx: gpu序号,默认为0
            mode: TensorRT模型精度,默认为'fp32'
            trt_int8_calibrator: TensorRT校准类实例,默认为None
        """
        # 初始化模型参数
        self.logger = logger
        self.onnx_model_path = onnx_model_path
        self.tensorrt_model_path = tensorrt_model_path
        self.output_shape = []
        self.mode = mode.lower()
        self.trt_int8_calibrator = trt_int8_calibrator
        self.trt_version = int(trt.__version__.split(".")[0])

        # 生成TensorRT模型文件
        if not os.path.exists(self.tensorrt_model_path):
            self.onnx2tensorrt()

        # 初始化TensorRT推理引擎相关变量
        self.device_ctx = cuda.Device(gpu_idx).make_context()
        self.engine = self.load_tensorrt_engine()
        self.context = self.engine.create_execution_context()
        self.input, self.output, self.bindings, self.stream = self.allocate_buffers(self.context)

    def __del__(self):
        del self.bindings
        del self.stream
        self.device_ctx.detach() # 2. 实例释放时需要detech cuda上下文
        del self.engine
        del self.context
        del self.input
        del self.output

    def onnx2tensorrt(self):
        """
        这是ONNX转TensorRT模型的函数
        Returns:
        """
        if self.trt_version >= 8:
            self.onnx2tensorrt8()
        else:
            self.onnx2tensorrt7()

    def onnx2tensorrt7(self):
        """
        这是利用ONNX模型生成TensorRT模型的函数,TensorRT7及以下版本接口
        Returns:
        """
        assert self.mode in ['fp32', 'fp16', 'int8'], "mode should be in ['fp32', 'fp16', 'int8'], " \
                                                 "but got {}".format(self.mode)

        trt_logger = trt.Logger(getattr(trt.Logger, 'ERROR'))
        builder = trt.Builder(trt_logger)

        self.logger.info("Loading ONNX file from path {}...".format(self.onnx_model_path))
        EXPLICIT_BATCH = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        network = builder.create_network(EXPLICIT_BATCH)
        parser = trt.OnnxParser(network, trt_logger)
        if isinstance(self.onnx_model_path, str):
            with open(self.onnx_model_path, 'rb') as f:
                self.logger.info("Beginning ONNX file parsing")
                flag = parser.parse(f.read())
        else:
            flag = parser.parse(self.onnx_model_path.read())
        if not flag:
            for error in range(parser.num_errors):
                self.logger.info(parser.get_error(error))

        self.logger.info("Completed parsing of ONNX file.")
        # re-order output tensor
        output_tensors = [network.get_output(i) for i in range(network.num_outputs)]
        [network.unmark_output(tensor) for tensor in output_tensors]
        for tensor in output_tensors:
            identity_out_tensor = network.add_identity(tensor).get_output(0)
            identity_out_tensor.name = 'identity_{}'.format(tensor.name)
            network.mark_output(tensor=identity_out_tensor)

        config = builder.create_builder_config()
        config.max_workspace_size = (1 << 25)
        if self.mode == 'fp16':
            assert builder.platform_has_fast_fp16, "not support fp16"
            builder.fp16_mode = True
        if self.mode == 'int8':
            assert builder.platform_has_fast_int8, "not support int8"
            builder.int8_mode = True
            builder.int8_calibrator = self.trt_int8_calibrator

        # if strict_type_constraints:
        #     config.set_flag(trt.BuilderFlag.STRICT_TYPES)

        self.logger.info("Building an engine from file {}; this may take a while...".format(self.onnx_model_path))
        engine = builder.build_cuda_engine(network)
        self.logger.info("Create engine successfully!")

        self.logger.info("Saving TRT engine file to path {}".format(self.tensorrt_model_path))
        with open(self.tensorrt_model_path, 'wb') as f:
            f.write(engine.serialize())
        self.logger.info("Engine file has already saved to {}!".format(self.tensorrt_model_path))

    def onnx2tensorrt8(self):
        """
        这是利用ONNX模型生成TensorRT模型的函数,TensorRT8及以上版本接口
        Returns:
        """
        assert self.mode in ['fp32', 'fp16', 'int8'], "mode should be in ['fp32', 'fp16', 'int8'], " \
                                                 "but got {}".format(self.mode)

        trt_logger = trt.Logger(getattr(trt.Logger, 'ERROR'))
        builder = trt.Builder(trt_logger)

        self.logger.info("Loading ONNX file from path {}...".format(self.onnx_model_path))
        EXPLICIT_BATCH = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        network = builder.create_network(EXPLICIT_BATCH)
        parser = trt.OnnxParser(network, trt_logger)
        if isinstance(self.onnx_model_path, str):
            with open(self.onnx_model_path, 'rb') as f:
                self.logger.info("Beginning ONNX file parsing")
                flag = parser.parse(f.read())
        else:
            flag = parser.parse(self.onnx_model_path.read())
        if not flag:
            for error in range(parser.num_errors):
                self.logger.info(parser.get_error(error))

        self.logger.info("Completed parsing of ONNX file.")
        # re-order output tensor
        output_tensors = [network.get_output(i) for i in range(network.num_outputs)]
        [network.unmark_output(tensor) for tensor in output_tensors]
        for tensor in output_tensors:
            identity_out_tensor = network.add_identity(tensor).get_output(0)
            identity_out_tensor.name = 'identity_{}'.format(tensor.name)
            network.mark_output(tensor=identity_out_tensor)

        config = builder.create_builder_config()
        config.max_workspace_size = (1 << 25)
        if self.mode == 'fp16':
            assert builder.platform_has_fast_fp16, "not support fp16"
            config.set_flag(trt.BuilderFlag.FP16)
        if self.mode == 'int8':
            assert builder.platform_has_fast_int8, "not support int8"
            config.set_flag(trt.BuilderFlag.INT8)
            config.int8_calibrator =self.trt_int8_calibrator

        # if strict_type_constraints:
        #     config.set_flag(trt.BuilderFlag.STRICT_TYPES)

        self.logger.info("Building an engine from file {}; this may take a while...".format(self.onnx_model_path))
        engine = builder.build_engine(network,config)
        self.logger.info("Create engine successfully!")

        self.logger.info("Saving TRT engine file to path {}".format(self.tensorrt_model_path))
        with open(self.tensorrt_model_path, 'wb') as f:
            f.write(engine.serialize())
        self.logger.info("Engine file has already saved to {}!".format(self.tensorrt_model_path))

    def load_tensorrt_engine(self):
        """
        这是加载TensorRT模型的函数
        Args:
            self:
        Returns:
        """
        if os.path.exists(self.tensorrt_model_path):
            TRT_LOGGER = trt.Logger()
            with open(self.tensorrt_model_path, "rb") as f, \
                    trt.Runtime(TRT_LOGGER) as runtime:
                engine = runtime.deserialize_cuda_engine(f.read())
            self.logger.info("Loaded TensorRT engine from file {}".format(self.tensorrt_model_path))
            return engine
        else:
            self.logger.error("TensorRT engine file {} does not exist!".format(self.tensorrt_model_path))
            sys.exit(1)

    def allocate_buffers(self, context):
        """
        这是为TensorRT推理引擎分配缓存的函数
        Args:
            context: 上下文信息
        Returns:
        """
        inputs = []
        outputs = []
        bindings = []
        stream = cuda.Stream()
        for binding in self.engine:
            #print(self.engine.get_binding_shape(binding))
            #size = trt.volume(self.engine.get_binding_shape(binding)) * self.engine.max_batch_size
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
            else:
                outputs.append(HostDeviceMem(host_mem, device_mem))
                binding_idx = self.engine.get_binding_index(binding)
                self.output_shape.append(self.engine.get_binding_shape(binding_idx))
        return inputs, outputs, bindings, stream

    def inference(self,input_tensor):
        """
        这是TensorRT推理引擎的前向推理的函数
        Args:
            input_tensor: 输入张量
        Returns:
        """
        # Copy data to input memory buffer
        [np.copyto(_inp.host, _input_tensor.ravel()) for _inp,_input_tensor in zip(self.input,input_tensor)]
        # Push to device
        self.device_ctx.push()
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
        # Pop the device
        self.device_ctx.pop()
        host_outputs = [out.host.reshape(output_shape) for out,output_shape in zip(self.output[::-1],self.output_shape)]
        return host_outputs

