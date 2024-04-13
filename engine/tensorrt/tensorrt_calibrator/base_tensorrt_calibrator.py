# -*- coding: utf-8 -*-
# @Time    : 2024/4/14 上午1:57
# @Author  : DaiPuWei
# @Email   : daipuwei@qq.com
# @File    : base_tensorrt_calibrator.py
# @Software: PyCharm

import os
import numpy as np
import pycuda.autoinit
import pycuda.driver as cuda
import tensorrt as trt

TRT_LOGGER = trt.Logger(trt.Logger.VERBOSE)

class TensorRTCalibrator(trt.IInt8EntropyCalibrator2):

    def __init__(self, calibration_dataloader, calibrator_table_path):
        """
        这是TensorRT模型INT8量化校准类的初始化函数
        Args:
            calibration_dataloader: 量化数据集加载器
            calibrator_table_path: 校准量化表文件路径
        """
        super(TensorRTCalibrator, self).__init__()
        # 初始化相关参数
        self.calibrator_table = os.path.abspath(calibrator_table_path)
        self.calibration_dataloader = calibration_dataloader
        self.device_input = cuda.mem_alloc(self.calibration_dataloader.get_calibration_data_size())

    def get_batch(self, names, p_str=None):
        """
        这是获取一个小批量数据的函数
        Args:
            names:
            p_str:
        Returns:
        """
        batch_input_tensor = self.calibration_dataloader.get_batch_data()
        if not batch_input_tensor.size:
            return None
        cuda.memcpy_htod(self.device_input, batch_input_tensor)
        return [int(self.device_input)]

    def get_batch_size(self):
        """
        这是获取batch_size的函数
        Returns:
        """
        return self.calibration_dataloader.get_batch_size()

    def read_calibration_cache(self):
        """
        这是读取校准表的函数
        Returns:
        """
        # If there is a cache, use it instead of calibrating again. Otherwise, implicitly return None.
        if os.path.exists(self.calibrator_table):
            with open(self.calibrator_table, "rb") as f:
                return f.read()

    def write_calibration_cache(self, cache):
        """
        这是将校准缓存写入校准表
        Args:
            cache:
        Returns:
        """
        with open(self.calibrator_table, "wb") as f:
            f.write(cache)


class CalibrationDataloader(object):

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
            self.is_nchw = False
            self.batch_size, self.height, self.width, self.channel = input_shape
            self.input_shape = (self.batch_size, self.height, self.width, self.channel)
        else:
            self.is_nchw = True
            self.batch_size, self.channel, self.height, self.width = input_shape
            self.input_shape = (self.batch_size, self.channel, self.height, self.width)
        self.calibrator_image_dir = os.path.abspath(calibrator_image_dir)

        # 初始化校准图片路径
        self.image_paths = []
        for image_name in os.listdir(self.calibrator_image_dir):
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
        """
        这是获取一个batch数据的函数
        Returns:
        """
        pass

    def get_batch_size(self):
        """
        这是获取batch_size的函数
        Returns:
        """
        return self.batch_size

    def get_calibration_data_size(self):
        """
        这是获取校准数据张量大小的函数
        Returns:
        """
        return self.calibration_data_size
