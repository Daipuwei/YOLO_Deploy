# -*- coding: utf-8 -*-
# @Time    : 2023/9/26 下午7:47
# @Author  : DaiPuWei
# @Email   : daipuwei@qq.com
# @File    : yolov5_trt_calibrator.py
# @Software: PyCharm


"""
    这是定义YOLOv5校准集数据加载器的脚本
"""

import cv2
import numpy as np
import pycuda.driver as cuda

from utils import letterbox
from engine.tensorrt import Calibration_Dataloader

class YOLOv5_Calibration_Dataloader(Calibration_Dataloader):

    def __init__(self,logger,input_shape,calibrator_image_dir,data_type='float32'):
        """
        这是YOLOv5模型INT8量化校准数据集加载器的初始化函数
        Args:
            logger: 日志类实例
            input_shape: 输入形状
            calibrator_image_dir: 校准图片集文件夹路径
            data_type: 数据类型,默认为'float32'
        """
        self.logger = logger
        super(YOLOv5_Calibration_Dataloader,self).__init__(input_shape=input_shape,
                                                           calibrator_image_dir=calibrator_image_dir,
                                                           data_type=data_type)
    def preprocess_image(self, image):
        """
        这是YOLOv5对单张图像进行预处理的函数
        Args:
            image: 图像，opencv格式
        Returns:
        """
        # 等比例缩放图像
        image_tensor = letterbox(image,(self.height,self.width))
        # BGR转RGB
        image_tensor = cv2.cvtColor(image_tensor, cv2.COLOR_BGR2RGB)
        # 归一化
        image_tensor = image_tensor / 255.0
        # hwc->chw
        image_tensor = np.transpose(image_tensor,(2, 0, 1))
        image_tensor = np.ascontiguousarray(image_tensor)
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
