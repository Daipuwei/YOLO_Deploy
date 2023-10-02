# -*- coding: utf-8 -*-
# @Time    : 2023/9/26 下午7:47
# @Author  : DaiPuWei
# @Email   : daipuwei@qq.com
# @File    : yolov5_trt_calibrator.py
# @Software: PyCharm

"""
    这是YOLOv5模型的TensorRT的INT8量化校准器定义脚本
"""

import cv2
import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit
import tensorrt as trt

from utils import letterbox
from engine import TensorRT_Calibrator
class YOLOv5_Calibrator(TensorRT_Calibrator):

    def __init__(self,logger,input_shape,calibrator_image_dir,calibrator_cache_path):
        """
        这是YOLOv5 TensorRT INT8量化校准类的初始化函数
        Args:
            logger: 日志类实例
            input_shape: 输入形状
            calibrator_image_dir: 校准图片集文件夹路径
            calibrator_cache_path: 校准缓存路径
        """
        super(YOLOv5_Calibrator,self).__init__(logger=logger,
                                               input_shape=input_shape,
                                               calibrator_image_dir=calibrator_image_dir,
                                               calibrator_cache_path=calibrator_cache_path)

    def preprocess_image(self, image):
        """
        这是YOLOv5对单张图像进行预处理的函数
        Args:
            image: 图像，opencv格式
        Returns:
        """
        # 填充像素并等比例缩放
        # print(np.shape(image))
        h, w = np.shape(image)[0:2]

        # image_tensor = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_tensor = letterbox(image, (self.height, self.width))
        # image_tensor = cv2.resize(image, (self.h, self.w))
        # cv2.imwrite("demo.jpg",image_tensor)
        image_tensor = np.transpose(image_tensor, (2, 0, 1))
        image_tensor = np.ascontiguousarray(image_tensor)
        # 归一化
        image_tensor = image_tensor / 255.0
        # 扩充维度
        #image_tensor = np.expand_dims(image_tensor,0)
        return image_tensor

    def next_batch(self):
        """
        这是获取一个小批量图像数据的函数
        Returns:
        """
        if self.batch_idx < self.max_batch_idx:
            batch_image_paths = self.image_paths[self.batch_idx * self.batch_size:(self.batch_idx + 1) * self.batch_size]
            batch_image_tensors = np.zeros((self.batch_size, self.channel, self.height, self.width),dtype=np.float32)
            for i,image_path in enumerate(batch_image_paths):
                image = cv2.imread(image_path)
                image = self.preprocess_image(image)
                assert (image.nbytes == self.data_size // self.batch_size), 'not valid img!' + image_path
                batch_image_tensors[i] = image
            self.batch_idx += 1
            self.logger.info("batch:[{}/{}]".format(self.batch_idx, self.max_batch_idx))
            return np.ascontiguousarray(batch_image_tensors)
        else:
            return np.array([])

    def get_batch(self, names, p_str=None):
        try:
            batch_image_tensors = self.next_batch()
            if batch_image_tensors.size == 0 or batch_image_tensors.size != self.batch_size * self.channel * self.height * self.width:
                return None
            cuda.memcpy_htod(self.device_input, batch_image_tensors.astype(np.float32))
            return [int(self.device_input)]
        except:
            return None