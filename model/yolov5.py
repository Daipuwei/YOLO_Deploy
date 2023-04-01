# -*- coding: utf-8 -*-
# @Time    : 2023/3/28 下午2:32
# @Author  : DaiPuWei
# @Email   : daipuwei@qq.com
# @File    : yolov5.py
# @Software: PyCharm

"""
    这是定义类YOLOv5模型的脚本
"""

import os
import sys
import cv2
import numpy as np

from model import DetectionModel
from utils import load_yaml
from utils import letterbox
from utils import get_classes
from utils import xywh2xyxy
from utils import scale_coords
from utils import draw_detection_results

class YOLOv5(DetectionModel):

    def __init__(self,logger,cfg,gpu_id=0):
        """
        这是YOLOv5的初始化函数
        Args:
            logger: 日志类实例
            cfg: 配置参数文件或者配置参数字典
            gpu_id: gpu设备号,默认为0
        """
        if isinstance(cfg, str):
            self.cfg = load_yaml(cfg)
        elif isinstance(cfg, dict):
            self.cfg = cfg
        else:
            self.logger.info("cfg must be a str or a dict")
            sys.exit(1)
        class_names = get_classes(os.path.abspath(cfg['DetectionModel']['class_names_path']))
        super(YOLOv5,self).__init__(logger=logger,
                                    class_names=class_names,
                                    onnx_model_path=cfg['DetectionModel']['model_path'],
                                    input_shape=cfg['DetectionModel']['input_shape'],
                                    model_type=cfg['DetectionModel']['model_type'],
                                    engine_type=cfg['DetectionModel']['engine_type'],
                                    engine_mode=cfg['DetectionModel']['mode'],
                                    gpu_id=gpu_id,
                                    confidence_threshold=cfg['DetectionModel']['confidence_threshold'],
                                    iou_threshold=cfg['DetectionModel']['iou_threshold'])
        self.logger.info("初始化YOLOv5检测模型成功")

    def preprocess_single_image(self,image):
        """
        这是YOLOv5对单张图像进行预处理的函数
        Args:
            image: 图像，opencv格式
        Returns:
        """
        # 填充像素并等比例缩放
        h,w = np.shape(image)[0:2]

        scale = np.array([w / self.w, h / self.h, w / self.w, h / self.h], dtype=np.float32)
        image_tensor = letterbox(image,(self.h,self.w))
        #cv2.imwrite("demo.jpg",image_tensor)
        image_tensor = np.transpose(image_tensor,(2, 0, 1))
        image_tensor = np.ascontiguousarray(image_tensor)
        # 归一化
        image_tensor = image_tensor / 255.0
        return image_tensor,scale,(h,w)

    def preprocess_batch_images(self,batch_images):
        """
        这是YOLOv5对批量图像进行处理的函数
        Args:
            batch_images: 批量图像数组，每张图像opencv读入
        Returns:
        """
        image_tensor = []
        scales = []
        image_shapes = []
        for image in batch_images:
            # 单独处理一张图像
            _image_tensor,_scale,_image_shape = self.preprocess_single_image(image)
            image_tensor.append(_image_tensor)
            scales.append(_scale)
            image_shapes.append(_image_shape)
        image_tensor = np.ascontiguousarray(image_tensor)
        scales = np.array(scales)
        image_shapes = np.array(image_shapes)
        return image_tensor,scales,image_shapes

    def preprocess(self,image):
        """
        这是YOLOv5的图像预处理函数
        Args:
            image: 输入图像，可以为单张图像也可以为图像数组
        Returns:
        """
        assert self.image_num > 0, "input image(s) must be 1 at least"
        assert self.image_num <= self.batchsize, "input image(s) size is {0} which is greater than model batch size:{1}".format(self.image_num,self.batchsize)
        image_tensor = np.zeros((self.batchsize, self.c, self.h, self.w),dtype=np.float32)
        scales = np.zeros((self.batchsize,4))
        image_shapes = np.zeros((self.batchsize,2))
        if isinstance(image,list):
            image = np.array(image)
        elif isinstance(image,np.ndarray):
            shape = np.shape(image)
            if len(shape) == 3:
                image = np.expand_dims(image,0)
        _image_tensor,_scale,_image_shape = self.preprocess_batch_images(image)
        image_tensor[0:self.image_num] = _image_tensor
        scales[0:self.image_num] = _scale
        image_shapes[0:self.image_num] = _image_shape
        return image_tensor,scales,image_shapes

    def postprocess_single_image(self,dets,image_shape):
        """
        这是对一张图像的预测结果进行后处理的函数
        Args:
            dets: 预测结果，shape为(anchor_num,6)
            scale: 放缩数组
            image_shape: 图像尺度
        Returns:
        """
        # 根据置信度对预测框进行过滤
        #print(np.shape(dets))
        obj_conf = dets[:, 4]
        # dets = dets[obj_conf > self.confidence_threshold]
        # obj_conf = obj_conf[obj_conf > self.confidence_threshold]
        # print(np.shape(obj_conf))

        # 计算每个类别的后验概率
        cls_id = np.argmax(dets[:, 5:],axis=1)
        # print(np.shape(cls_id))
        scores = np.array([dets[i, 5+id] for i,id in enumerate(cls_id)])
        # print(np.shape(scores))
        scores *= obj_conf
        bboxes = dets[..., 0:4]

        # 利用置信度进一步对预测框进行过过滤
        # valid_scores = scores > self.confidence_threshold
        # if len(valid_scores) == 0:
        #     return []
        # cls_id = cls_id[valid_scores]
        # scores = scores[valid_scores]
        # bboxes = bboxes[valid_scores]
        #print(bboxes)

        # 对预测框进行坐标格式进行转换，并还原到原始尺度
        bboxes = xywh2xyxy(bboxes)
        bboxes = scale_coords(bboxes, (self.h, self.w), image_shape)

        # 使用NMS算法过滤冗余框
        indices = cv2.dnn.NMSBoxes(bboxes.tolist(), scores.tolist(),
                                   self.confidence_threshold,self.iou_threshold)
        indices = np.reshape(indices,(len(indices),))
        if len(indices) == 0:
            return []
        bboxes = bboxes[indices]
        #print(bboxes)
        scores = scores[indices].reshape((-1,1))
        cls_id = cls_id[indices].reshape((-1,1))
        preds = np.concatenate([bboxes,scores,cls_id],axis=1)
        return preds

    def postprocess(self,outputs,image_shapes):
        """
        这是YOLOv5模型后处理函数
        Args:
            outputs: 模型输出结果张量,shape为(batchsize,anchor_num,6)
            scales: 图像放缩系数数组，shape为(batchsize,4)
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

    def detect(self,image):
        """
        这是YOLO-FaceV2进行人脸检测的函数
        Args:
            image: 输入图像，可以为单张图像也可以为图像数组
        Returns:
        """
        # 获取图像个数
        if isinstance(image,list):
            self.image_num = len(image)
        elif isinstance(image,np.ndarray):
            shape = np.shape(image)
            if len(shape) == 3:
                self.image_num = 1
            else:
                self.image_num = shape[0]

        # 做图像预处理
        image_tensor,scales,image_shapes = self.preprocess(image)
        # 模型推理
        if self.engine is None:
            return []
        outputs = self.engine.inference([image_tensor])[0]
        if outputs is None:
            return []
        # 对推理结果进行后处理
        outputs = self.postprocess(outputs,image_shapes)

        if self.image_num == 1:         # 仅有一张图像，则进行降维
            outputs = outputs[0]
            for x1, y1, x2, y2, score, cls_id in outputs:
                x1 = int(round(x1))
                y1 = int(round(y1))
                x2 = int(round(x2))
                y2 = int(round(y2))
                cls_id = int(cls_id)
                self.logger.info("检测到{0}, bbox: {1},{2},{3},{4},"
                                 "score:{5}".format(self.class_names[cls_id],x1,y1,x2,y2,score))
        else:
            for i in range(self.image_num):
                for x1, y1, x2, y2, score, cls_id in outputs[i]:
                    x1 = int(round(x1))
                    y1 = int(round(y1))
                    x2 = int(round(x2))
                    y2 = int(round(y2))
                    cls_id = int(cls_id)
                    self.logger.info(
                        "检测到{0},bbox: {1},{2},{3},{4},"
                        "score:{5}".format(self.class_names[cls_id], x1, y1, x2, y2,score))
        return outputs

    def detect_video(self,video_path,result_dir,interval=-1):
        """
        这是检测视频的函数
        Args:
            video_path: 视频路径
            result_dir: 检测结果保存文件夹路径
            interval: 视频抽帧频率,默认为-1,逐帧检测
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
        dir,video_name = os.path.split(video_path)
        fname,ext = os.path.splitext(video_name)
        video_result_path = os.path.join(result_dir,fname+"_result"+ext)
        vid_writer = cv2.VideoWriter(video_result_path,cv2.VideoWriter_fourcc(*'mp4v'),fps, (w, h))

        # 遍历视频，逐帧进行检测
        cnt = -1
        while True:
            return_value, frame = vid_cap.read()
            cnt += 1
            #print(return_value)
            if return_value:
                if interval == -1:      #  逐帧检测
                    preds = self.detect(frame)
                    if len(preds) == 0:
                        continue
                    detect_image = draw_detection_results(frame, preds, self.class_names, self.colors)
                    vid_writer.write(detect_image)
                else:                   # 间隔ineterval秒检测
                    if cnt % (interval*fps) == 0:
                        preds = self.detect(frame)
                        if len(preds) == 0:
                            continue
                        detect_image = draw_detection_results(frame, preds, self.class_names, self.colors)
                        vid_writer.write(detect_image)
            else:
                break
        vid_cap.release()
        vid_writer.release()
