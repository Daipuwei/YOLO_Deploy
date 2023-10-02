# -*- coding: utf-8 -*-
# @Time    : 2023/3/28 下午2:37
# @Author  : DaiPuWei
# @Email   : daipuwei@qq.com
# @File    : tensorrt_calibrator.py
# @Software: PyCharm


"""
    这是定义TensorRT校准类脚本
"""

import os
import sys
import cv2
import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit
import tensorrt as trt
#cuda.init()

class TensorRT_Calibrator(trt.IInt8EntropyCalibrator2):

    def __init__(self,logger,input_shape,calibrator_image_dir,calibrator_cache_path):
        """
        这是TensorRT模型抽象INT8量化校准类的初始化函数
        Args:
            logger: 日志类实例
            input_shape: 输入形状
            calibrator_image_dir: 校准图片集文件夹路径
            calibrator_cache_path: 校准表缓存文件路径
        """
        # 初始化相关参数
        self.logger = logger
        self.batch_size,self.channel,self.height,self.width = input_shape
        self.cache_file = os.path.abspath(calibrator_cache_path)
        self.calibrator_image_dir = os.path.abspath(calibrator_image_dir)

        # 初始化校准图片路径
        self.image_paths = []
        for image_name in os.listdir(self.calibrator_image_dir):
            self.image_paths.append(os.path.join(self.calibrator_image_dir,image_name))

        self.batch_idx = 0
        self.max_batch_idx = len(self.image_paths) // self.batch_size
        self.data_size = self.batch_size * self.channel * self.height * self.width * trt.float32.itemsize
        self.device_input = cuda.mem_alloc(self.data_size)

    def next_batch(self):
        pass

    def get_batch(self, names, p_str=None):
        pass

    def get_batch_size(self):
        return self.batch_size

    def read_calibration_cache(self):
        # If there is a cache, use it instead of calibrating again. Otherwise, implicitly return None.
        if os.path.exists(self.cache_file):
            with open(self.cache_file, "rb") as f:
                return f.read()

    def write_calibration_cache(self, cache):
        with open(self.cache_file, "wb") as f:
            f.write(cache)