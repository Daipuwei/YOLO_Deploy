# -*- coding: utf-8 -*-
# @Time    : 2023/12/26 下午3:13
# @Author  : DaiPuWei
# @Email   : daipuwei@qq.com
# @File    : tensorrt_utils.py
# @Software: PyCharm

"""
    这是定义TensorRT相关工具函数及其工具类的脚本
"""

import os
import numpy as np
import pycuda.autoinit
import pycuda.driver as cuda
import tensorrt as trt

TRT_LOGGER = trt.Logger(trt.Logger.VERBOSE)


# TRT_LOGGER = trt.Logger()
class TensorRT_Calibrator(trt.IInt8EntropyCalibrator2):

    def __init__(self, calibration_dataloader, calibrator_table_path):
        """
        这是TensorRT模型INT8量化校准类的初始化函数
        Args:
            calibration_dataloader: 量化数据集加载器
            calibrator_table_path: 校准量化表文件路径
        """
        super(TensorRT_Calibrator, self).__init__()
        # 初始化相关参数
        self.calibrator_table = os.path.abspath(calibrator_table_path)
        self.calibration_dataloader = calibration_dataloader
        self.device_input = cuda.mem_alloc(self.calibration_dataloader.get_calibration_data_size())

    def get_batch(self, names, p_str=None):
        # print("================================")
        # print("images")
        # print("================================")
        batch_input_tensor = self.calibration_dataloader.get_batch_data()
        # print("**************")
        # print(batch_input_tensor)
        # print(batch_input_tensor.size)
        # print("**************")
        if not batch_input_tensor.size:
            return None
        cuda.memcpy_htod(self.device_input, batch_input_tensor)
        return [int(self.device_input)]

    def get_batch_size(self):
        return self.calibration_dataloader.get_batch_size()

    def read_calibration_cache(self):
        # If there is a cache, use it instead of calibrating again. Otherwise, implicitly return None.
        if os.path.exists(self.calibrator_table):
            with open(self.calibrator_table, "rb") as f:
                return f.read()

    def write_calibration_cache(self, cache):
        with open(self.calibrator_table, "wb") as f:
            f.write(cache)


class Calibration_Dataloader():

    def __init__(self, input_shape, calibrator_image_dir, data_type='float32'):
        """
        这是校准数据集加载器的初始化函数
        Args:
            input_shape: 输入尺度
            calibrator_image_dir: 校准图像文件夹路径
            data_type: 数据类型，默认为"float32"
        """
        # 初始化相关变量
        if input_shape[1] > 3:
            self.batch_size, self.height, self.width, self.channel = input_shape
            self.input_shape = (self.batch_size, self.height, self.width, self.channel)
        else:
            self.batch_size, self.channel, self.height, self.width = input_shape
            self.input_shape = (self.batch_size, self.channel, self.height, self.width)
        self.calibrator_image_dir = os.path.abspath(calibrator_image_dir)

        # 初始化校准图片路径
        self.image_paths = []
        if os.path.exists(calibrator_image_dir) or len(list(os.listdir(calibrator_image_dir))):
            for image_name in os.listdir(self.calibrator_image_dir):
                self.image_paths.append(os.path.join(self.calibrator_image_dir, image_name))
        for image_name in os.listdir(self.calibrator_image_dir):
            # print(image_name)
            # print(os.path.join(self.calibrator_image_dir,image_name))
            self.image_paths.append(os.path.join(self.calibrator_image_dir, image_name))
        self.image_paths = np.array(self.image_paths)
        self.logger.info("Find {} images in Calibration Dataset".format(len(self.image_paths)))

        # 初始化相关变量
        self.batch_idx = 0
        self.max_batch_idx = len(self.image_paths) // self.batch_size
        self.image_num = self.max_batch_idx * self.batch_size
        self.data_type = data_type
        if self.data_type == 'float32':
            self.calibration_data = np.zeros(self.input_shape, dtype=np.float32)
        else:
            self.calibration_data = np.zeros(self.input_shape, dtype=np.float16)
        self.calibration_data_size = self.calibration_data.nbytes

    def __len__(self):
        return self.max_batch_idx

    def get_batch_data(self):
        pass

    def get_batch_size(self):
        return self.batch_size

    def get_calibration_data_size(self):
        return self.calibration_data_size


def onnx2tensorrt(logger, onnx_model_path, tensorrt_model_path, mode, calibrator=None):
    """
    这是ONNX转TensorRT的函数
    Args:
        logger: 日志实例
        onnx_model_path: ONNX模型路径
        tensorrt_model_path: TensorRT模型路径
        mode: TensorRT模型类型，候选值有['fp32','fp16','int8']
        calibrator: INT8校准类实例，默认为None
    Returns:
    """
    trt_version = int(trt.__version__.split(".")[0])
    if trt_version >= 8:
        onnx2tensorrtv8(logger, onnx_model_path, tensorrt_model_path, mode, calibrator)
    else:
        onnx2tensorrtv7(logger, onnx_model_path, tensorrt_model_path, mode, calibrator)


def onnx2tensorrtv7(logger, onnx_model_path, tensorrt_model_path, mode, calibrator=None):
    """
    这是ONNX转TensorRT的函数,支持TensorRT7.x及其更低版本
    Args:
        logger: 日志实例
        onnx_model_path: ONNX模型路径
        tensorrt_model_path: TensorRT模型路径
        mode: TensorRT模型类型，候选值有['fp32','fp16','int8']
        calibrator: INT8校准类实例，默认为None
    Returns:
    """
    assert mode in ['fp32', 'fp16', 'int8'], "mode should be in ['fp32', 'fp16', 'int8'], " \
                                             "but got {}".format(mode)
    # 初始化相关变量
    network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    # if mode == "int8" and calibrator is None:
    #     network_flags = network_flags | (1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_PRECISION))
    with trt.Builder(TRT_LOGGER) as builder, builder.create_network(flags=network_flags) as network, \
            trt.OnnxParser(network, TRT_LOGGER) as parser:
        with open(onnx_model_path, 'rb') as model:
            # 解析ONNX
            logger.info("Loading ONNX file from path {}...".format(onnx_model_path))
            if not parser.parse(model.read()):
                logger.info('ERROR: ONNX Parse Failed')
                for error in range(parser.num_errors):
                    logger.error(parser.get_error(error))
                    return
            logger.info("Completed parsing of ONNX file.")

            # 设置生成TensorRT模型的相关配置
            config = builder.create_builder_config()
            config.max_workspace_size = (1 << 30)
            if mode == 'fp16':
                assert builder.platform_has_fast_fp16, "not support fp16"
                builder.fp16_mode = True
            if mode == 'int8':
                assert builder.platform_has_fast_int8, "not support int8"
                builder.int8_mode = True
                if calibrator is not None:
                    builder.int8_calibrator = calibrator

            # 生成TensorRT模型引擎并完成序列化保存为文件
            logger.info("Building an engine from file {}; this may take a while...".format(onnx_model_path))
            engine = builder.build_cuda_engine(network)
            logger.info("Create engine successfully!")
            logger.info("Saving TRT engine file to path {}".format(tensorrt_model_path))
            with open(tensorrt_model_path, 'wb') as f:
                f.write(engine.serialize())
            logger.info("Engine file has already saved to {}!".format(tensorrt_model_path))


def onnx2tensorrtv8(logger, onnx_model_path, tensorrt_model_path, mode, calibrator=None):
    """
    这是ONNX转TensorRT的函数,支持TensorRT8.x及其更高版本
    Args:
        logger: 日志实例
        onnx_model_path: ONNX模型路径
        tensorrt_model_path: TensorRT模型路径
        mode: TensorRT模型类型，候选值有['fp32','fp16','int8']
        calibrator: INT8校准类实例，默认为None
    Returns:
    """
    assert mode in ['fp32', 'fp16', 'int8'], "mode should be in ['fp32', 'fp16', 'int8'], " \
                                             "but got {}".format(mode)
    # 初始化相关变量
    network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    if mode == "int8" and calibrator is None:
        network_flags = network_flags | (1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_PRECISION))
    with trt.Builder(TRT_LOGGER) as builder, builder.create_network(flags=network_flags) as network, \
            trt.OnnxParser(network, TRT_LOGGER) as parser:
        with open(onnx_model_path, 'rb') as model:
            # 解析ONNX
            logger.info("Loading ONNX file from path {}...".format(onnx_model_path))
            if not parser.parse(model.read()):
                logger.info('ERROR: ONNX Parse Failed')
                for error in range(parser.num_errors):
                    logger.error(parser.get_error(error))
                    return None
            logger.info("Completed parsing of ONNX file.")

            # 设置生成TensorRT模型的相关配置
            config = builder.create_builder_config()
            config.max_workspace_size = (1 << 30)
            if mode == 'fp16':
                assert builder.platform_has_fast_fp16, "not support fp16"
                # config.set_flag(trt.BuilderFlag.FP16)
                config.flags |= 1 << int(trt.BuilderFlag.FP16)
            if mode == 'int8':
                assert builder.platform_has_fast_int8, "not support int8"
                # config.set_flag(trt.BuilderFlag.INT8)
                config.flags |= 1 << int(trt.BuilderFlag.INT8)
                config.flags |= 1 << int(trt.BuilderFlag.FP16)
                if calibrator is not None:
                    config.int8_calibrator = calibrator

            # 生成TensorRT模型引擎并完成序列化保存为文件
            logger.info("Building an engine from file {}; this may take a while...".format(onnx_model_path))
            engine = builder.build_engine(network, config)
            logger.info("Create engine successfully!")
            logger.info("Saving TRT engine file to path {}".format(tensorrt_model_path))
            with open(tensorrt_model_path, 'wb') as f:
                f.write(engine.serialize())
            logger.info("Engine file has already saved to {}!".format(tensorrt_model_path))
