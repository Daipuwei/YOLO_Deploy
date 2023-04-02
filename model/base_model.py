# -*- coding: utf-8 -*-
# @Time    : 2023/3/28 下午2:30
# @Author  : DaiPuWei
# @Email   : daipuwei@qq.com
# @File    : base_model.py
# @Software: PyCharm

"""
    这是定义检测模型基础类的函数
"""

import os
import colorsys

class DetectionModel:

    def __init__(self, logger, onnx_model_path, class_names,input_shape,
                 model_type="yolov5", engine_type='onnx',engine_mode="fp32",
                 confidence_threshold=0.5, iou_threshold=0.5,
                 gpu_id=0, calibrator_image_dir=None, calibrator_cache_path=None):
        """
        这是抽象检测模型类的初始化函数
        Args:
            logger: 日志类实例
            onnx_model_path: onnx模型文件路径
            class_names: 目标分类名称数组
            model_type: 模型类型,默认为'yolov5'
            engine_type: 推理引擎类型，默认为'onnx'
            engine_mode: 推理引擎模式，默认为'fp32'
            input_name: 输入节点名称
            output_name: 输出节点名称
            input_shape: 输入shape
            output_shape: 输出shape
            confidence_threshold: 置信度阈值，默认为0.5
            iou_threshold: iou阈值，默认为0.5
            gpu_id:gpu设备号,默认为0
            calibrator_image_dir: 校准图像文件夹路径,默认为None
            calibrator_cache_path: 校准缓存文件路径,默认为None
        """
        # 初始化模型参数
        self.logger = logger
        self.onnx_model_path = os.path.abspath(onnx_model_path)
        self.class_names = class_names
        self.model_type = model_type
        self.engine_type = engine_type.lower()
        self.engine_mode = engine_mode.lower()
        self.input_shape = input_shape
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.batchsize, self.c, self.h, self.w = input_shape
        self.image_num = 0
        self.gpu_id = gpu_id
        self.calibrator_image_dir = calibrator_image_dir
        self.calibrator_cache_path = calibrator_cache_path

        # 初始化不同引擎的模型路径
        dir,model_name = os.path.split(self.onnx_model_path)
        fname,ext = os.path.splitext(model_name)
        if self.engine_type == 'onnx':
            self.engine_model_path = self.onnx_model_path
        elif self.engine_type == 'tensorrt':
            self.engine_model_path = os.path.join(dir,fname+".trt")
        else:
            self.engine_model_path = self.onnx_model_path

        # 初始化颜色列表
        hsv_tuples = [(x / len(self.class_names), 1., 1.)
                      for x in range(len(self.class_names))]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), self.colors))

        # 初始化推理引擎
        if self.engine_type == 'onnx':
            from .engine import ONNX_Engine
            self.engine = ONNX_Engine(logger=logger,
                                      onnx_model_path=self.engine_model_path)
        elif self.engine_type == 'tensorrt':
            from .engine import TensorRT_Engine
            if self.engine_mode == 'int8' or self.calibrator_cache_path is not None \
                    or self.calibrator_image_dir is not None:
                if self.model_type == "yolov5":
                    from utils import YOLOv5_Calibrator
                    trt_int8_calibrator = YOLOv5_Calibrator(logger=logger,
                                                            input_shape=self.input_shape,
                                                            calibrator_image_dir=self.calibrator_image_dir,
                                                            calibrator_cache_path=self.calibrator_cache_path)
                else:
                    from utils import YOLOv5_Calibrator
                    trt_int8_calibrator = YOLOv5_Calibrator(logger=logger,
                                                            input_shape=self.input_shape,
                                                            calibrator_image_dir=self.calibrator_image_dir,
                                                            calibrator_cache_path=self.calibrator_cache_path)
            else:
                trt_int8_calibrator = None
            self.engine = TensorRT_Engine(logger=logger,
                                          onnx_model_path=self.onnx_model_path,
                                          tensorrt_model_path=self.engine_model_path,
                                          gpu_idx=self.gpu_id,
                                          mode=self.engine_mode,
                                          trt_int8_calibrator=trt_int8_calibrator)
        else:
            from model.engine.onnx_model import ONNX_Engine
            self.engine = ONNX_Engine(logger=logger,
                                      onnx_model_path=self.engine_model_path)

    def get_batch_size(self):
        return self.batchsize

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

    def detect(self,image,export_time=False):
        """
        这是检测的图像函数
        Args:
            image: 输入图像，可以为单张图像也可以为图像数组
            export_time: 是否输出时间信息标志位，默认为False
        Returns:
        """
        pass

    def detect_video(self,video_path,result_dir,interval=-1):
        """
        这是检测视频的函数
        Args:
            video_path: 视频路径
            result_dir: 结果保存文件夹路径
            interval: 视频抽帧频率,默认为-1,逐帧检测
        Returns:
        """
        pass
