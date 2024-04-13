# -*- coding: utf-8 -*-
# @Time    : 2023/3/28 下午2:30
# @Author  : DaiPuWei
# @Email   : daipuwei@qq.com
# @File    : base_model.py
# @Software: PyCharm

"""
    这是定义检测模型基础类的脚本
"""

from utils.detection_utils import random_generate_colors

class DetectionModel(object):

    def __init__(self,logger,engine,class_names,model_type="yolov5",
                 confidence_threshold=0.5,iou_threshold=0.5,**kwargs):
        """
        这是抽象检测模型类的初始化函数
        Args:
            logger: 日志类实例
            engine: 推理引擎实例
            class_names: 目标分类名称数组
            model_type: 模型类型,默认为'yolov5'
            confidence_threshold: 置信度阈值，默认为0.5
            iou_threshold: iou阈值，默认为0.5
        """
        # 初始化模型参数
        self.logger = logger
        self.class_names = class_names
        self.model_type = model_type
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.image_num = 0

        # 初始化推理引擎
        self.engine = engine

        # 初始化颜色列表
        self.colors = random_generate_colors(len(self.class_names))

        # 初始化模型输入shape
        self.input_shape = self.engine.get_input_shape()[0]
        if self.input_shape[1] <= 3:
            self.is_nchw = True
            self.batch_size, self.c, self.h, self.w = self.input_shape
        else:
            self.is_nchw = False
            self.batch_size, self.h, self.w, self.c = self.input_shape

    def get_batch_size(self):
        return self.batch_size

    def get_class_names(self):
        return self.class_names

    def get_image_num(self):
        return self.image_num

    def get_colors(self):
        return self.colors

    def preprocess(self, image):
        """
        这是检测模型的图像预处理函数
        Args:
            image: 输入图像，可以为单张图像也可以为图像数组
        Returns:
        """
        pass

    def postprocess(self,results):
        """
        这是检测模型的后处理函数
        Args:
            results: 模型输出结果张量
        Returns:
        """
        pass

    def detect(self,image,export_time=False,print_detection_result=False):
        """
        这是检测的图像函数
        Args:
            image: 输入图像，可以为单张图像也可以为图像数组
            export_time: 是否输出时间标志位,默认为False
            print_detection_result：是否打印检测结果，默认为False
        Returns:
        """
        pass

    def detect_video(self,video_path,video_result_path,interval=-1,print_detection_result=False):
        """
        这是检测视频的函数
        Args:
            video_path: 视频路径
            video_result_path: 检测结果视频路径
            interval: 视频抽帧频率,默认为-1,逐帧检测
            print_detection_result：是否打印检测结果，默认为False
        Returns:
        """
        pass
