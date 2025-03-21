# -*- coding: utf-8 -*-
# @Time    : 2025/3/21 10:30
# @Author  : DaiPuWei
# @Email   : puwei.dai@vitalchem.com
# @File    : ppyoloe.py
# @Software: PyCharm

"""
    这是定义PPYOLOE模型的脚本
"""

import os
import cv2
import time
import numpy as np

from model import DetectionModel
from model import MODEL_REGISTRY
from engine import build_engine

from utils import letterbox
from utils import get_classes
from utils import xywh2xyxy
from utils import scale_coords
from utils import draw_detection_results
from utils import logger_config

class PPYOLOE(DetectionModel):

    def __init__(self,logger,cfg,gpu_id=0,**kwargs):
        """
        这是PPYOLOE的初始化函数
        Args:
            logger: 日志类实例
            cfg: 配置参数文件或者配置参数字典
            gpu_id: gpu设备号,默认为0
        """
        class_names = get_classes(os.path.abspath(cfg['class_name_path']))
        engine = build_engine(logger,cfg, gpu_id=gpu_id)
        super(PPYOLOE,self).__init__(logger=logger,
                                    engine=engine,
                                    class_names=class_names,
                                    model_type=cfg['model_type'],
                                    confidence_threshold=cfg['confidence_threshold'],
                                    iou_threshold=cfg['iou_threshold'],
                                    gpu_id=gpu_id,
                                    **kwargs)
        self.logger.info("初始化PPYOLOE检测模型成功")

    def preprocess_single_image(self,image):
        """
        这是PPYOLOE对单张图像进行预处理的函数
        Args:
            image: 图像，opencv格式
        Returns:
        """
        # 填充像素并等比例缩放
        h,w = np.shape(image)[0:2]
        image_tensor = letterbox(image,(self.h,self.w))
        image_tensor = image_tensor.astype(np.float32)
        # BGR转RGB
        image_tensor = cv2.cvtColor(image_tensor, cv2.COLOR_BGR2RGB)
        # # 归一化
        # image_tensor = image_tensor / 255.0
        # # hwc->chw
        # if self.engine.get_is_nchw():
        #     image_tensor = np.transpose(image_tensor, (2, 0, 1))
        image_tensor = np.ascontiguousarray(image_tensor)
        return image_tensor,(h,w)

    def preprocess_batch_images(self,batch_images):
        """
        这是PPYOLOE对批量图像进行处理的函数
        Args:
            batch_images: 批量图像数组，每张图像opencv读入
        Returns:
        """
        image_tensor = []
        image_shapes = []
        for image in batch_images:
            # 单独处理一张图像
            _image_tensor,_image_shape = self.preprocess_single_image(image)
            image_tensor.append(_image_tensor)
            image_shapes.append(_image_shape)
        image_tensor = np.ascontiguousarray(image_tensor)
        image_shapes = np.array(image_shapes)
        return image_tensor,image_shapes

    def preprocess(self,image):
        """
        这是PPYOLOE的图像预处理函数
        Args:
            image: 输入图像，可以为单张图像也可以为图像数组
        Returns:
        """
        assert self.image_num > 0, "input image(s) must be 1 at least"
        assert self.image_num <= self.batch_size, "input image(s) size is {0} which is greater than model batch size:{1}".format(self.image_num,self.batch_size)
        if self.is_nchw:
            image_tensor = np.zeros((self.batch_size, self.c, self.h, self.w),dtype=np.float32)
        else:
            image_tensor = np.zeros((self.batch_size, self.h, self.w, self.c), dtype=np.float32)
        image_shapes = np.zeros((self.batch_size,2))
        _image_tensor,_image_shape = self.preprocess_batch_images(image)
        image_tensor[0:self.image_num] = _image_tensor
        image_shapes[0:self.image_num] = _image_shape
        return image_tensor,image_shapes

    def postprocess_single_image(self,dets,image_shape):
        """
        这是PPYOLOE对一张图像的预测结果进行后处理的函数
        Args:
            dets: 模型输出结果张量,shape为(1,bbox_num，num_classes+5)
            image_shapes: 图像尺度数组，shape为(1,2)
        Returns:
        """
        # 预测框进行过过滤
        obj_conf = dets[:, 4]
        mask = obj_conf > self.confidence_threshold
        if len(mask) == 0:
            return []
        dets = dets[mask]

        # 对检测结果进行解码解码
        cls_id = np.argmax(dets[:, 5:], axis=1)
        scores = np.max(dets[:, 5:], axis=1)
        bboxes = dets[..., 0:4]

        # 对预测框进行坐标格式进行转换，并还原到原始尺度
        bboxes = scale_coords(bboxes, (self.h, self.w), image_shape)

        # 使用NMS算法过滤冗余框
        indices = cv2.dnn.NMSBoxes(bboxes.tolist(), scores.tolist(),
                                   self.confidence_threshold, self.iou_threshold)
        indices = np.reshape(indices, (len(indices),))
        if len(indices) == 0:
            return []
        bboxes = bboxes[indices]
        scores = scores[indices].reshape((-1, 1))
        cls_id = cls_id[indices].reshape((-1, 1))
        preds = np.concatenate([bboxes, scores, cls_id], axis=1)

        # 对结果进行编码
        outputs = []
        for x1, y1, x2, y2, score, cls_id in preds:
            x1 = round(x1)
            y1 = round(y1)
            x2 = round(x2)
            y2 = round(y2)
            cls_id = int(cls_id)
            score = round(score, 4)
            outputs.append({"bbox": [x1, y1, x2, y2],
                            "score": score,
                            "cls_id": cls_id,
                            "cls_name": self.class_names[cls_id]})
        return outputs

    def postprocess(self,outputs,image_shapes):
        """
        这是PPYOLOE模型后处理函数
        Args:
            outputs: 模型输出结果张量,shape为(batchsize,anchor_num*(num_classes+4))
            image_shapes: 图像尺度数组，shape为(batchsize,2)
        Returns:
        """
        preds = []
        #print(np.shape(outputs))
        for i in np.arange(self.image_num):
            _image_shape = image_shapes[i]
            dets = outputs[i]

            # 对预测框进行后处理
            pred = self.postprocess_single_image(dets,_image_shape)
            if len(pred) == 0:
                preds.append([])
                #self.logger.info("该帧图像未检测到目标")
            else:
                preds.append(pred)
                #self.logger.info("该帧图像检测到{0}个目标".format(len(pred)))
        return preds

    def detect(self,image,export_time=False,print_detection_result=False):
        """
        这是PPYOLOE模型检测图像的函数
        Args:
            image: 输入图像，可以为单张图像也可以为图像数组
            export_time: 是否输出时间信息标志位，默认为False
            print_detection_result: 是否打印检测结果,默认为Fasle
        Returns:
        """
        # 获取图像个数
        if isinstance(image,list):
            self.image_num = len(image)
            image = np.array(image)
        elif isinstance(image,np.ndarray):
            shape = np.shape(image)
            if len(shape) == 3:
                self.image_num = 1
                image = np.expand_dims(image, 0)
            else:
                self.image_num = shape[0]

        # 做图像预处理
        preprocess_start = time.time()
        image_tensor,image_shapes = self.preprocess(image)
        preprocess_end = time.time()
        preprocess_time = (preprocess_end-preprocess_start)*1000
        # 模型推理
        if self.engine is None:
            return []
        inference_start = time.time()
        outputs = self.engine.inference([image_tensor])[0]
        inference_end = time.time()
        inference_time = (inference_end-inference_start)*1000
        if outputs is None:
            return []
        # 对推理结果进行后处理
        postprocess_start = time.time()
        outputs = self.postprocess(outputs,image_shapes)
        postprocess_end = time.time()
        postprocess_time = (postprocess_end-postprocess_start)*1000
        detect_time = preprocess_time+inference_time+postprocess_time
        self.logger.info("预处理时间：{0:.2f}ms,推理时间:{1:.2f}ms,后处理时间：{2:.2f}ms,检测时间：{3:.2f}ms"
                         .format(round(preprocess_time,2),round(inference_time,2),
                                 round(postprocess_time,2),round(detect_time,2)))

        if self.image_num == 1:         # 仅有一张图像，则进行降维
            outputs = outputs[0]
            if print_detection_result:
                for output in outputs:
                    x1,y1,x2,y2 = output['bbox']
                    score = output['score']
                    cls_name = output['cls_name']
                    x1 = int(round(x1))
                    y1 = int(round(y1))
                    x2 = int(round(x2))
                    y2 = int(round(y2))
                    self.logger.info("检测到{0}, bbox: {1},{2},{3},{4},"
                                     "score:{5:.4f}".format(cls_name, x1, y1, x2, y2, score))
        else:
            if print_detection_result:
                for i in range(self.image_num):
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
        if export_time:
            return outputs, {"preprocess_time": preprocess_time,
                             "inference_time": inference_time,
                             "postprocess_time": postprocess_time,
                             "detect_time": detect_time}
        else:
            return outputs

    def detect_video(self,video_path,video_result_path,interval=-1,print_detection_result=False):
        """
        这是检测视频的函数
        Args:
            video_path: 视频路径
            video_result_path: 检测结果视频路径
            interval: 视频抽帧频率,默认为-1,逐帧检测
            print_detection_result: 是否打印检测结果，默认为False
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
        vid_writer = cv2.VideoWriter(video_result_path,cv2.VideoWriter_fourcc(*'mp4v'),fps, (w, h))

        # 遍历视频，逐帧进行检测
        cnt = -1
        while True:
            return_value, frame = vid_cap.read()
            cnt += 1
            #print(return_value)
            if return_value:
                if interval == -1:      #  逐帧检测
                    preds = self.detect(frame,False,print_detection_result)
                    if len(preds) == 0:
                        continue
                    detect_image = draw_detection_results(frame, preds, self.colors)
                    vid_writer.write(detect_image)
                else:                   # 间隔ineterval秒检测
                    if cnt % (interval*fps) == 0:
                        preds = self.detect(frame)
                        if len(preds) == 0:
                            continue
                        detect_image = draw_detection_results(frame, preds, self.colors)
                        vid_writer.write(detect_image)
            else:
                break
        vid_cap.release()
        vid_writer.release()

@MODEL_REGISTRY.register()
def ppyoloe(logger,cfg,**kwargs):
    """
    这是PPYOLOE的初始化函数
    Args:
        logger: 日志类实例
        cfg: 参数配置字典
        **kwargs: 自定义参数
    Returns:
    """
    model = PPYOLOE(logger,cfg,**kwargs)
    return model
