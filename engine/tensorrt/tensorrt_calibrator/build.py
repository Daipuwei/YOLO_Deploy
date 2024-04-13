# -*- coding: utf-8 -*-
# @Time    : 2024/4/14 上午1:54
# @Author  : DaiPuWei
# @Email   : daipuwei@qq.com
# @File    : build.py
# @Software: PyCharm

"""
    这是定义TensorRT推理引擎INT8校准集加载器注册脚本
"""

from utils import Registry

TENSORRT_CALIBRATION_DATALOADER_REGISTRY = Registry("TENSORRT_CALIBRATION_DATALOADER")
TENSORRT_CALIBRATION_DATALOADER_REGISTRY.__doc__ = """这是TensorRT推理引擎的INT8校准集加载器的注册器，用于自动生成不同模型INT8校准集加载器"""

def build_tensorrt_calibration_dataloader(logger,input_shape,calibrator_image_dir,data_type,calibrator_table_path,model_type):
    """
    这是TensorRT校准集加载器的注册函数
    Args:
        logger: 日志类实例
        input_shape: 模型输入尺度
        calibrator_image_dir: 校准集文件夹路径
        data_type: 数据类型
        calibrator_table_path: 校准表路径
        model_type:模型类型
    Returns:
    """
    trt_calibrator_type = model_type+"_trt_calibrator"
    return TENSORRT_CALIBRATION_DATALOADER_REGISTRY.get(trt_calibrator_type)(logger,input_shape,calibrator_image_dir,data_type,calibrator_table_path)