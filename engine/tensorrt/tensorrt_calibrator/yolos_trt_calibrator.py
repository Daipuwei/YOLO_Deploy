# -*- coding: utf-8 -*-
# @Time    : 2024/3/22 下午9:23
# @Author  : DaiPuWei
# @Email   : daipuwei@qq.com
# @File    : yolos_trt_calibrator.py
# @Software: PyCharm

"""
    这是定义YOLOS校准集数据加载器的脚本
"""

import cv2
import numpy as np

from .build import TENSORRT_CALIBRATION_DATALOADER_REGISTRY
from .base_tensorrt_calibrator import TensorRTCalibrator,CalibrationDataloader

class YOLOSCalibrationDataloader(CalibrationDataloader):

    def __init__(self,logger,input_shape,calibrator_image_dir,data_type='float32'):
        """
        这是YOLOS模型INT8量化校准数据集加载器的初始化函数
        Args:
            logger: 日志类实例
            input_shape: 输入形状
            calibrator_image_dir: 校准图片集文件夹路径
            data_type: 数据类型,默认为'float32'
        """
        self.logger = logger
        self.mean = np.array([0.485, 0.456, 0.406])
        self.std = np.array([0.229, 0.224, 0.225])
        super(YOLOSCalibrationDataloader,self).__init__(input_shape=input_shape,
                                                        calibrator_image_dir=calibrator_image_dir,
                                                        data_type=data_type)
    def preprocess_image(self, image):
        """
        这是YOLOS对单张图像进行预处理的函数
        Args:
            image: 图像，opencv格式
        Returns:
        """
        # 填充像素并等比例缩放
        image_tensor = cv2.resize(image,(self.width,self.height), interpolation=cv2.INTER_LINEAR)
        image_tensor = image_tensor.astype(np.float32)
        # BGR转RGB
        image_tensor = cv2.cvtColor(image_tensor, cv2.COLOR_BGR2RGB)
        # 归一化
        image_tensor = image_tensor / 255.0
        image_tensor -= self.mean
        image_tensor /= self.std
        # hwc->chw
        if self.is_nchw:
            image_tensor = np.transpose(image_tensor,(2, 0, 1))
        return image_tensor

    def get_batch_data(self):
        """
        这是获取一个小批量图像数据的函数
        Returns:
        """
        if self.batch_idx == self.max_batch_idx:
            self.batch_idx = 0
            return np.array([])
        else:
            batch_image_paths = self.image_paths[self.batch_idx*self.batch_size:(self.batch_idx+1)*self.batch_size]
            for i,image_path in enumerate(batch_image_paths):
                image = cv2.imread(image_path)
                image_tensor = self.preprocess_image(image)
                self.calibration_data[i] = image_tensor
            self.logger.info("batch:[{}/{}]".format(self.batch_idx, self.max_batch_idx))
            self.batch_idx += 1
            return self.calibration_data

@TENSORRT_CALIBRATION_DATALOADER_REGISTRY.register()
def yolos_trt_calibrator(logger,input_shape,calibrator_image_dir,data_type,calibrator_table_path):
    """
    这是YOLOS的TensorRT推理引擎INT8校准集加载器的注册函数
    Args:
        logger: 日志类实例
        input_shape: 模型输入尺寸
        calibrator_image_dir: 校准集文件夹路径
        data_type: 数据类型
        calibrator_table_path: 校准表路径
    Returns:
    """
    calibration_dataloader = YOLOSCalibrationDataloader(logger,input_shape,calibrator_image_dir,data_type)
    trt_calibrator = TensorRTCalibrator(calibration_dataloader,calibrator_table_path)
    return trt_calibrator