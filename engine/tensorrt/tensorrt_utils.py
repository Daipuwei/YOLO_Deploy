# -*- coding: utf-8 -*-
# @Time    : 2023/12/26 下午3:13
# @Author  : DaiPuWei
# @Email   : daipuwei@qq.com
# @File    : tensorrt_utils.py
# @Software: PyCharm

"""
    这是定义TensorRT相关工具函数及其工具类的脚本
"""

import pycuda.autoinit
import pycuda.driver as cuda
import tensorrt as trt

TRT_LOGGER = trt.Logger(trt.Logger.VERBOSE)
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
