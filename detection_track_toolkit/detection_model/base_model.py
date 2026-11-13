# -*- coding: utf-8 -*-
# @Time    : 2026/9/9 09:07
# @Author  : DaiPuWei
# @Email   : daipuwei@qq.com
# @File    : base_model.py
# @Software: PyCharm

"""
    这是定义检测模型基类的脚本。

    模块内容：
        1. DetectionModel：检测模型基类，三块装配（推理引擎+预处理器+后处理器）
           的编排器：
           - 构造时装配engine/preprocessor/postprocessor三块并校验，
             公共属性class_names/num_classes/双阈值/colors与布局元信息
             （batch_size/c/h/w/is_nchw）从各块读取对外暴露；
           - detect(image, export_time, print_detection_result)：图像检测骨架
             （批次数为局部变量：输入归一化→不满批补零图凑满→预处理→引擎推理
             →按真实图像数后处理→打印/计时）；
           - detect_video(video_path, ...)：视频检测骨架（无检测帧也写入原帧
             不丢帧，抽帧模式下非采样帧写原帧；写器创建失败显式报错；
             检测循环异常时经try/finally释放读流与写流）；
           - release()：资源释放契约，级联释放推理引擎；
           - kwargs逐键注入为模型属性（下划线开头或覆盖已有属性抛ValueError）。

        子类：model.anchor_based.yolov5.YOLOv5（anchorbase家族）、
              model.dflyolo.dflyolo.DFLYOLO（DFL家族）、
              model.yolox.yolox.YOLOX（自成体系），
        均由 model.build.MODEL_REGISTRY 注册工厂函数，由 build_model 按名称装配。
"""

import time

import cv2
import numpy as np

from .detection_utils import draw_detection_results
from .detection_utils import random_generate_colors

class DetectionModel(object):

    def __init__(self,logger,engine,preprocessor,postprocessor,
                 model_type="yolov5",**kwargs):
        """
        这是检测模型基类的初始化函数（三块装配形态）
        Args:
            logger: 日志类实例
            engine: 推理引擎实例
            preprocessor: 预处理器实例
            postprocessor: 后处理器实例
            model_type: 模型类型,默认为'yolov5'
            kwargs: 自定义参数字典（逐键注入为模型属性；键不能以下划线开头，
                    也不能与已有属性同名，否则抛ValueError）
        """
        # 参数校验：三块与日志实例不能为空
        if logger is None:
            raise ValueError("日志实例不能为None")
        if engine is None:
            raise ValueError("推理引擎实例不能为None")
        if preprocessor is None or postprocessor is None:
            raise ValueError("preprocessor与postprocessor必须同时提供")
        self.logger = logger
        self.engine = engine
        self.preprocessor = preprocessor
        self.postprocessor = postprocessor
        self.model_type = model_type
        self.class_names = postprocessor.class_names
        if self.class_names is None or len(self.class_names) == 0:
            raise ValueError("类别名称列表不能为空")
        self.num_classes = len(self.class_names)
        self.confidence_threshold = postprocessor.confidence_threshold
        self.iou_threshold = postprocessor.iou_threshold
        self.colors = random_generate_colors(self.num_classes)
        # 布局元信息由预处理器对外暴露
        self.batch_size = preprocessor.batch_size
        self.c = preprocessor.c
        self.h = preprocessor.h
        self.w = preprocessor.w
        self.is_nchw = preprocessor.is_nchw
        # 其余kwargs注入为属性：逐键检查，拒绝覆盖已有属性（拼错键/冲突键立即报错，
        # 而不是静默接受后在使用处才炸）
        for key, value in kwargs.items():
            if key.startswith("_") or hasattr(self, key):
                raise ValueError("非法或冲突的检测模型配置键'{0}'：不能以下划线开头或覆盖已有属性"
                                 .format(key))
            setattr(self, key, value)

    def get_batch_size(self):
        """
        这是获取模型批量大小的函数
        Returns:
            模型批量大小
        """
        return self.batch_size

    def get_class_names(self):
        """
        这是获取目标分类名称数组的函数
        Returns:
            目标分类名称数组
        """
        return self.class_names

    def get_colors(self):
        """
        这是获取目标分类颜色列表的函数
        Returns:
            目标分类颜色列表
        """
        return self.colors

    def release(self):
        """
        这是释放检测模型资源的函数（统一级联释放三块装配：
        推理引擎/预处理器/后处理器；脚本侧统一调用本方法即可，
        各块的release按需覆写，缺省空实现）
        """
        if self.engine is not None:
            self.engine.release()
        if self.preprocessor is not None:
            self.preprocessor.release()
        if self.postprocessor is not None:
            self.postprocessor.release()

    def print_detection_results(self,outputs,image_num):
        """
        这是打印检测结果的函数
        Args:
            outputs: 检测结果列表（每张图像一个条目）
            image_num: 图像数量
        """
        for i in range(image_num):
            for output in outputs[i]:
                x1,y1,x2,y2 = output['bbox']
                score = output['score']
                cls_name = output['cls_name']
                x1 = int(round(x1))
                y1 = int(round(y1))
                x2 = int(round(x2))
                y2 = int(round(y2))
                self.logger.info("检测到{0}, bbox: {1},{2},{3},{4},"
                                 "score:{5:.4f}".format(cls_name, x1, y1, x2, y2, score))

    def detect(self,image,export_time=False,print_detection_result=False):
        """
        这是检测的图像函数。
        输入图像数小于模型批量大小时自动补零图凑满（本工具链仅支持固定批量
        引擎，预处理块已校验输入shape各维为正整数）：补批行仅占位，推理后
        按真实图像数（image_shapes[:image_num]）做后处理，补批行的输出被丢弃
        不进入检测结果——数据集尾批不满batchsize与视频单帧检测（batch>1模型）
        由此支持；输入图像数超过批量大小时报错
        Args:
            image: 输入图像，可以为单张图像也可以为图像数组
            export_time: 是否输出时间标志位,默认为False
            print_detection_result：是否打印检测结果，默认为False
        Returns:
            检测结果（export_time为True时附带时间信息字典）
        """
        # 获取图像个数（局部变量，不做实例状态残留；非法输入显式报错）
        if isinstance(image,(list, tuple)):
            image_num = len(image)
            images = np.array(image)
        elif isinstance(image,np.ndarray):
            shape = np.shape(image)
            if len(shape) == 3:
                image_num = 1
                images = np.expand_dims(image, 0)
            elif len(shape) == 4:
                image_num = shape[0]
                images = image
            else:
                raise ValueError("输入图像必须为3维(H,W,C)或4维(batchsize,H,W,C)数组，"
                                 "实际shape为{0}".format(shape))
        else:
            raise ValueError("输入图像必须为list/tuple或numpy数组，实际类型为{0}"
                             .format(type(image).__name__))
        if image_num > self.batch_size:
            raise ValueError("输入图像数量{0}超过模型批量大小{1}"
                             .format(image_num, self.batch_size))
        if image_num < self.batch_size:
            # 固定批量引擎不满批时补零图凑满：补批行仅占位，
            # 推理后postprocess按真实图像数循环，补批行的输出被丢弃。
            # （修复：数据集尾批不满batchsize与视频单帧检测在batch>1的
            # 固定批量引擎上曾直接崩溃——TRT元素数校验ValueError、
            # ORT输入维度不匹配报错；本工具链仅支持固定batch引擎，
            # 预处理块已校验输入shape各维为正整数，无按实际批量推理的选项）
            pad = np.zeros((self.batch_size - image_num,) + np.shape(images)[1:],
                           dtype=images.dtype)
            images = np.concatenate([images, pad], axis=0)
        # 做图像预处理（返回批量张量、逐图原始尺度与逐图缩放逆变换参数
        # ——restore_params即逐图w/h的缩放系数与偏移系数，三块契约三元组）
        preprocess_start = time.perf_counter()
        image_tensor,image_shapes,restore_params = self.preprocessor.preprocess(images)
        preprocess_end = time.perf_counter()
        preprocess_time = (preprocess_end-preprocess_start)*1000
        # 模型推理
        inference_start = time.perf_counter()
        outputs = self.engine.inference([image_tensor])
        inference_end = time.perf_counter()
        inference_time = (inference_end-inference_start)*1000
        if outputs is None:
            return []
        # 单输出模型取[0]；多输出模型保留全部输出列表交后处理
        # （后处理器经multioutputs_decode统一解码）
        if len(outputs) == 1:
            outputs = outputs[0]
        # 对推理结果进行后处理（只处理真实图像，不满批补零的占位行被丢弃；
        # restore_params同步裁剪，坐标还原按预处理给出的逐图缩放系数与偏移系数）
        postprocess_start = time.perf_counter()
        image_shapes = image_shapes[:image_num]
        restore_params = restore_params[:image_num]
        outputs = self.postprocessor.postprocess(outputs,image_shapes,restore_params)
        postprocess_end = time.perf_counter()
        postprocess_time = (postprocess_end-postprocess_start)*1000
        detect_time = preprocess_time+inference_time+postprocess_time
        # 计时日志按export_time门控（视频逐帧检测时避免日志风暴）
        if export_time:
            self.logger.info("预处理时间：{0:.2f}ms,推理时间:{1:.2f}ms,后处理时间：{2:.2f}ms,检测时间：{3:.2f}ms"
                             .format(round(preprocess_time,2),round(inference_time,2),
                                     round(postprocess_time,2),round(detect_time,2)))
        if print_detection_result:
            self.print_detection_results(outputs, image_num)

        if image_num == 1:         # 仅有一张图像，则进行降维
            outputs = outputs[0]
        if export_time:
            return outputs, {"preprocess_time": preprocess_time,
                             "inference_time": inference_time,
                             "postprocess_time": postprocess_time,
                             "detect_time": detect_time}
        else:
            return outputs

    def detect_video(self,video_path,video_result_path,interval=-1,
                    print_detection_result=False,export_time=False):
        """
        这是检测视频的函数
        Args:
            video_path: 视频路径
            video_result_path: 检测结果视频路径
            interval: 视频抽帧频率,默认为-1,逐帧检测
            print_detection_result：是否打印检测结果，默认为False
            export_time: 是否导出检测时间，默认为False
                （True时逐帧记录预处理/推理/后处理耗时日志）
        Returns:
        """
        # 初始化输入视频
        vid_cap = cv2.VideoCapture(video_path)
        if not vid_cap.isOpened():
            raise IOError("Couldn't open webcam or video")
        fps = int(round(vid_cap.get(cv2.CAP_PROP_FPS)))  # 视频的fps
        w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        # 初始化检测结果视频
        vid_writer = cv2.VideoWriter(video_result_path,cv2.VideoWriter_fourcc(*'mp4v'),fps,(w,h))
        if not vid_writer.isOpened():
            vid_cap.release()
            raise IOError("无法创建检测结果视频{0}（编码器不支持该尺寸）".format(video_result_path))
        # 抽帧步长：interval秒采样一次；fps异常时逐帧。
        # 步长下限钳到1（修复：interval*fps<1时int截断为0，
        # 曾导致cnt%0 ZeroDivisionError中断检测）
        step = max(1, int(interval * fps)) if (interval > 0 and fps > 0) else 1
        # 遍历视频，逐帧进行检测（无检测帧也写入原帧不丢帧，
        # 抽帧模式下非采样帧写原帧）；try/finally保证检测异常时
        # 读流与写流均被释放，不泄漏文件句柄
        try:
            cnt = -1
            while True:
                return_value, frame = vid_cap.read()
                cnt += 1
                if not return_value:
                    break
                detect_image = frame
                if interval < 0:        # 逐帧检测
                    preds = self.detect(frame,export_time,print_detection_result)
                    if export_time:
                        preds = preds[0]
                    if len(preds) > 0:
                        detect_image = draw_detection_results(frame, preds, self.colors)
                elif cnt % step == 0:   # 间隔interval秒检测
                    preds = self.detect(frame,export_time,print_detection_result)
                    if export_time:
                        preds = preds[0]
                    if len(preds) > 0:
                        detect_image = draw_detection_results(frame, preds, self.colors)
                vid_writer.write(detect_image)
        finally:
            vid_cap.release()
            vid_writer.release()
