#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2026/9/18
# @Author  : DaiPuWei
# @Email   : daipuwei@qq.com
# @File    : yolov5_onnxruntime_demo.py
# @Software: PyCharm

"""
    这是YOLOv5 ONNXRuntime部署推理测试脚本（自包含单文件交付版，
    不依赖detection_track_toolkit，仅需numpy/opencv-python/onnxruntime）。

    模型约定：
        - 输入：1x3xHxW（NCHW）BGR图像；等比例缩放填充（letterbox）、
          BGR转RGB、除以255归一化、HWC转CHW均在类内完成；
        - 输出：默认3输出模型（各检测层原始输出，每层shape为
          (1, num_anchors*(5+num_classes), H/s, W/s)，s=8/16/32，
          通道布局为逐anchor的[x,y,w,h,obj,cls...]，即检测头卷积输出布局），
          同时兼容单输出模型（融合输出(1,框数,5+num_classes)，
          列布局[cx,cy,w,h,obj,cls...]）；
        - 默认约定ONNX输出已带sigmoid激活（导出时对检测层整体激活，
          对应with_sigmoid=True），未带时传with_sigmoid=False由类内补做；
        - anchors：coco默认anchor，在类初始化中写死为(3,3,2)数组
          （shape为(层数,每层anchor数,宽高)，与模型训练配置一致即可，
          不一致时检测框尺寸会整体偏移）；
        - 各检测层stride写死为[8,16,32]（与模型结构一致）。

    使用方式：
        python yolov5_onnxruntime_demo.py --model ./yolov5_multioutputs.onnx \\
            --source ./test.jpg [--names ./coco.names] [--confidence_threshold 0.6] \\
            [--iou_threshold 0.45] [--result_dir ./result/detect] [--interval -1] \\
            [--print_detection_result]

    说明：source支持图像/视频的单文件或目录（递归遍历，按后缀自动分流）；
    结果统一归入result_dir/<模型文件名>/<时间戳>/下的image/与video/子目录
    （图像结果统一保存为jpg，视频结果统一保存为mp4，与detect.py一致）。
"""

import argparse
import colorsys
import os

import cv2
from datetime import datetime
import numpy as np
import onnxruntime as ort

# 图像文件后缀（与detect.py一致）
IMG_FORMATS = ['.bmp', '.dng', '.jpeg', '.jpg', '.mpo', '.png', '.tif', '.tiff', '.webp']
# 视频文件后缀（与detect.py一致）
VID_FORMATS = ['.asf', '.avi', '.gif', '.m4v', '.mkv', '.mov', '.mp4', '.mpeg', '.mpg', '.ts', '.wmv', '.dav']

# ONNXRuntime张量类型字符串到numpy dtype的映射表（输入输出OrtValue装配用）
TENSOR_TYPE_TO_NP = {
    "tensor(float)": np.float32,
    "tensor(float16)": np.float16,
    "tensor(int32)": np.int32,
    "tensor(int64)": np.int64,
    "tensor(uint8)": np.uint8,
}


def sigmoid(x):
    """
    这是sigmoid激活函数（向量化广播执行，对数组整体一次激活，
    多输出解码中ONNX未带sigmoid时使用）
    Args:
        x: 数值或数组（原始logit）
    Returns:
        sigmoid激活后的数值或数组
    """
    return 1.0 / (1.0 + np.exp(-x))


def make_grid(h,w,offset=0):
    """
    这是生成网格坐标的函数（(2,grids)布局：首行x坐标、次行y坐标，
    与检测层输出(C,grids)布局对齐，解码时免转置）
    Args:
        h: 网格高度（检测层输出特征图高）
        w: 网格宽度
        offset: 网格坐标整体偏移量，默认为0（如DFL解码传0.5得网格中心点）
    Returns:
        网格坐标数组，shape为(2,h*w)，首行为x、次行为y（网格单元单位，
        含offset整体偏移，不含stride缩放，缩放由调用方按stride执行）
    """
    yv, xv = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
    return np.stack((xv.reshape(-1), yv.reshape(-1)), axis=0).astype(np.float32) + offset


def resize_keep_aspect_ratio(image,resize_shape=(640, 640),color=(114, 114, 114)):
    """
    这是等比例缩放并填充像素的函数（letterbox）
    Args:
        image: 图像，opencv格式
        resize_shape: 目标尺寸(height,width)，默认为(640,640)
        color: 填充像素值，默认为(114,114,114)
    Returns:
        等比例缩放并填充到目标尺寸后的图像
    """
    h, w, _ = np.shape(image)
    input_h, input_w = resize_shape
    if h == input_h and w == input_w:
        return image
    # 计算等比例缩放比例与填充量
    gain = min(input_w / w, input_h / h)
    new_unpad = int(round(w * gain)), int(round(h * gain))
    top, bottom = 0, int(round((input_h - new_unpad[1]) + 0.1))
    left, right = 0, int(round((input_w - new_unpad[0]) + 0.1))
    resize_image = cv2.resize(image, new_unpad, interpolation=cv2.INTER_LINEAR)
    resize_image = cv2.copyMakeBorder(resize_image, top, bottom, left, right,
                                      cv2.BORDER_CONSTANT, value=color)
    return resize_image


def rescale_coords(preds,model_hw,image_shape):
    """
    这是检测框坐标从模型输入尺度还原到原始图像尺度的函数
    （与项目scale_coords一致：letterbox填充仅在右下，坐标为0起点，
    只需除以gain缩放；越界坐标裁剪到图像范围内）
    Args:
        preds: 预测数组（[x1,y1,x2,y2,score,cls_id]）
        model_hw: 模型输入尺寸(height,width)
        image_shape: 原始图像尺度(height,width)
    Returns:
        尺度还原后的预测数组
    """
    gain = min(model_hw[0] / image_shape[0], model_hw[1] / image_shape[1])
    preds[:, 0:4] /= gain
    preds[:, 0] = np.clip(preds[:, 0], 0, image_shape[1])   # x1
    preds[:, 1] = np.clip(preds[:, 1], 0, image_shape[0])   # y1
    preds[:, 2] = np.clip(preds[:, 2], 0, image_shape[1])   # x2
    preds[:, 3] = np.clip(preds[:, 3], 0, image_shape[0])   # y2
    return preds


def nms(preds,iou_threshold):
    """
    这是贪心NMS的函数（类别无关：所有类别混在一起按分数降序抑制重叠框）
    Args:
        preds: 预测数组（[x1,y1,x2,y2,score,cls_id]，
            bbox统一为x1y1x2y2格式）
        iou_threshold: IoU抑制阈值（IoU超过该值则抑制低分框）
    Returns:
        NMS过滤后的预测数组（bbox保持x1y1x2y2格式）
    """
    # 一次性IoU匹配：向量化计算全部两两IoU矩阵（NxN），
    # 后续贪心循环仅做矩阵查表，避免逐轮重复计算IoU
    boxes = preds[:, :4]
    xx1 = np.maximum(boxes[:, None, 0], boxes[None, :, 0])
    yy1 = np.maximum(boxes[:, None, 1], boxes[None, :, 1])
    xx2 = np.minimum(boxes[:, None, 2], boxes[None, :, 2])
    yy2 = np.minimum(boxes[:, None, 3], boxes[None, :, 3])
    inter = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
    area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    union = area[:, None] + area[None, :] - inter
    ious = inter / np.maximum(union, 1e-7)              # (N,N)两两IoU矩阵
    # 贪心循环：按分数降序依次保留最高分框，
    # 与保留框IoU超过阈值的候选经矩阵查表一次性剔除
    order = np.argsort(-preds[:, 4])
    keep = []
    while len(order) > 0:
        idx = order[0]
        keep.append(idx)
        if len(order) == 1:
            break
        order = order[1:][ious[idx, order[1:]] <= iou_threshold]
    return preds[keep]


def random_generate_colors(color_num):
    """
    这是随机生成RGB颜色数组的函数
    Args:
        color_num: 颜色个数
    Returns:
        rgb颜色列表
    """
    hsv_tuples = [(x / color_num, 1., 1.) for x in range(color_num)]
    rgb_colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
    rgb_colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
                          rgb_colors))
    return rgb_colors


def draw_detection_results(image,detecion_outputs,colors):
    """
    这是绘制检测结果的函数
    Args:
        image: 图像，opencv读入
        detecion_outputs: 检测结果字典数组（bbox为x1y1x2y2格式）
        colors: rgb颜色列表（按类别id索引）
    Returns:
        绘制检测结果后的图像
    """
    h, w, _ = np.shape(image)
    tl = max(round((h + w) // 300), 1)  # line/font thickness
    for output in detecion_outputs:
        x1, y1, x2, y2 = output['bbox']
        score = output['score']
        cls_name = output['cls_name']
        cls_id = output['cls_id']
        x1 = int(round(x1))
        y1 = int(round(y1))
        x2 = int(round(x2))
        y2 = int(round(y2))
        text = "{0}:{1:.4f}".format(cls_name, score)
        # 绘制检测框
        cv2.rectangle(image, (x1, y1), (x2, y2), colors[cls_id],
                      thickness=tl, lineType=cv2.LINE_AA)
        # 初始化标签字符串宽高及其坐标
        tf = max(tl - 1, 1)
        text_w, text_h = cv2.getTextSize(text, 0, fontScale=tl / 3, thickness=tf)[0]
        outside = y1 - text_h >= 3
        text_x1, text_y1 = x1, y1
        text_x2, text_y2 = x1 + text_w, y1 - text_h - 3 if outside else y1 + text_h + 3
        mean_color = np.mean(colors[cls_id])
        if mean_color > 128:
            text_color = (0, 0, 0)
        else:
            text_color = (255, 255, 255)
        # 绘制标签字符串
        cv2.rectangle(image, (text_x1, text_y1), (text_x2, text_y2),
                      colors[cls_id], thickness=-1, lineType=cv2.LINE_AA)
        cv2.putText(image, text, (x1, y1 - 2 if outside else y1 + text_h + 2),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=tl / 3,
                    color=text_color, thickness=tf, lineType=cv2.LINE_AA)
    return image


class YOLOv5ONNXRuntime(object):
    """
    这是YOLOv5 ONNXRuntime部署推理类（默认3输出多输出模型，兼容单输出模型；
    类内完成letterbox等比例缩放、BGR转RGB、除以255归一化、HWC转CHW预处理，
    以及多输出解码/单输出解码、NMS与坐标还原）
    """

    # 各检测层stride（与模型结构一致，3输出模型按此顺序解码）
    STRIDES = [8, 16, 32]

    def __init__(self,model_path,class_names,confidence_threshold=0.6,
                 iou_threshold=0.45,is_normalize=True,with_sigmoid=True,
                 gpu_id=0):
        """
        这是YOLOv5 ONNXRuntime推理类的初始化函数
        Args:
            model_path: ONNX模型文件路径（默认3输出多输出模型，兼容单输出模型）
            class_names: 类别名列表（顺序与类别id一致）
            confidence_threshold: 置信度阈值，默认为0.6
                （过滤规则：先按obj过滤、再按类别最大分过滤，同一阈值）
            iou_threshold: NMS抑制阈值，默认为0.45
            is_normalize: 是否做除以255归一化，默认为True
            with_sigmoid: ONNX输出是否已带sigmoid激活，默认为True
                （多输出模型检测层输出整体已激活时不再处理；
                False时多输出解码对整层输出经sigmoid一次广播激活）
            gpu_id: GPU设备号，默认为0（CUDA可用时OrtValue在指定GPU上分配）
        """
        # 检测参数
        self.class_names = list(class_names)
        self.num_classes = len(self.class_names)
        # 每类别随机生成一种RGB颜色（绘制检测结果用）
        self.colors = random_generate_colors(self.num_classes)
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.is_normalize = is_normalize
        self.with_sigmoid = with_sigmoid
        # anchors写死为coco默认anchor（(3层, 3anchor, 2宽高)数组，
        # 与模型训练配置不一致时检测框尺寸会整体偏移，需同步修改）
        self.anchors = np.array([
            [[10, 13], [16, 30], [33, 23]],
            [[30, 61], [62, 45], [59, 119]],
            [[116, 90], [156, 198], [373, 326]]], dtype=np.float32)  # (3,3,2)
        # 初始化ONNXRuntime会话（默认使用CUDA：CUDAExecutionProvider实际可用时
        # 走CUDA，否则自动回退CPU并告警）
        self.gpu_id = gpu_id
        self.is_cuda = "CUDAExecutionProvider" in ort.get_available_providers()
        if not self.is_cuda:
            print("[WARN] CUDAExecutionProvider不可用，自动回退CPU推理")
        # 初始化ONNXRuntime会话（会话选项与provider配置与项目
        # ONNXRuntimeEngine保持一致：图优化全开、顺序执行、单流推理；
        # CUDA使用HEURISTIC卷积算法选择与kSameAsRequested arena策略，
        # 保证与工具集推理数值完全一致）
        session_option = ort.SessionOptions()
        session_option.log_severity_level = 3          # ORT自身日志级别（3=ERROR）
        session_option.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        session_option.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        session_option.inter_op_num_threads = 1        # 算子间并行线程数（单流推理）
        if self.is_cuda:
            cuda_opts = {
                "device_id": self.gpu_id,
                "arena_extend_strategy": "kSameAsRequested",
            }
            # 版本防御：cudnn_conv_algo_search是ORT 1.13+的选项
            ort_major, ort_minor = (int(v) for v in ort.__version__.split('.')[:2])
            if (ort_major, ort_minor) >= (1, 13):
                cuda_opts["cudnn_conv_algo_search"] = "HEURISTIC"
            providers = [("CUDAExecutionProvider", cuda_opts), "CPUExecutionProvider"]
        else:
            providers = ["CPUExecutionProvider"]
        self.session = ort.InferenceSession(model_path, sess_options=session_option,
                                            providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        input_shape = self.session.get_inputs()[0].shape
        self.input_type = self.session.get_inputs()[0].type
        # 输入布局自动判断：NCHW（第2维为3）或NHWC（第4维为3），
        # NHWC模型无需HWC转CHW
        self.is_nchw = not (len(input_shape) == 4
                            and isinstance(input_shape[3], int)
                            and input_shape[3] == 3)
        # 校验模型输入shape并派生batch_size/c/h/w（与项目预处理器一致：
        # 仅支持4维且各维为正整数，动态维度提前明确报错）
        if len(input_shape) != 4:
            raise ValueError("模型输入shape必须为4维（NCHW或NHWC），实际为{0}"
                             .format(input_shape))
        if self.is_nchw:
            self.batch_size, self.c, self.h, self.w = input_shape
        else:
            self.batch_size, self.h, self.w, self.c = input_shape
        for dim in (self.batch_size, self.c, self.h, self.w):
            if not isinstance(dim, int) or dim <= 0:
                raise ValueError("模型输入shape各维必须为正整数，实际为{0}"
                                 .format(input_shape))
        self.model_hw = (self.h, self.w)
        self.output_names = [output.name for output in self.session.get_outputs()]
        self.output_shapes = [output.shape for output in self.session.get_outputs()]
        self.output_types = [output.type for output in self.session.get_outputs()]
        # 输出节点个数决定模型形态：多于1路为多输出模型，等于1路为单输出融合模型
        self.multi_output = len(self.output_names) > 1
        # IO Binding可用性判断（ONNXRuntime>=1.8提供OrtValue预分配API）
        ort_version = [int(v) for v in ort.__version__.split(".")[:2]]
        self.use_io_binding = ort_version[0] > 1 or \
            (ort_version[0] == 1 and ort_version[1] >= 8)
        # 预分配静态shape输出的OrtValue（仅各维均为正整数的输出可预分配；
        # 预分配缓冲在推理间复用，消除逐帧cudaMalloc/cudaFree与malloc开销，
        # 与ONNXRuntimeEngine保持一致）
        self.outputs = {}
        if self.use_io_binding:
            for output_name, output_shape, output_type in zip(
                    self.output_names, self.output_shapes, self.output_types):
                if all(isinstance(dim, int) and dim > 0 for dim in output_shape):
                    dtype = TENSOR_TYPE_TO_NP.get(output_type, np.float32)
                    if self.is_cuda:
                        self.outputs[output_name] = \
                            ort.OrtValue.ortvalue_from_shape_and_type(
                                output_shape, dtype, "cuda", self.gpu_id)
                    else:
                        self.outputs[output_name] = \
                            ort.OrtValue.ortvalue_from_shape_and_type(
                                output_shape, dtype)

    def preprocess(self,images):
        """
        这是批量图像预处理函数（letterbox等比例缩放填充→BGR转RGB→除以255
        归一化→HWC转CHW→stack为批量张量），类内完成hwc2chw与归一化
        Args:
            images: 图像数组，shape为(batch,H,W,C)（BGR）
        Returns:
            (输入张量, 原始图像尺度数组)：
            输入张量shape为(batch,3,H,W)的float32数组（NCHW模型）
            或(batch,H,W,3)（NHWC模型）；尺度数组shape为(batch,2)，
            每行为(原始高,原始宽)，后处理坐标还原用
        """
        # 图像数量与批量大小校验（不超过模型batch维，与项目一致）
        if len(images) > self.batch_size:
            raise ValueError("输入图像数量{0}超过模型批量大小{1}"
                             .format(len(images), self.batch_size))
        batch_tensors = []
        image_shapes = []
        for image in images:
            h, w = np.shape(image)[:2]
            image_tensor = resize_keep_aspect_ratio(image, self.model_hw)
            # 颜色转换：BGR转RGB（与训练侧一致）
            image_tensor = cv2.cvtColor(image_tensor, cv2.COLOR_BGR2RGB)
            # 归一化：除以255（is_normalize=False时跳过，归一化已融入ONNX）
            image_tensor = image_tensor.astype(np.float32)
            if self.is_normalize:
                image_tensor /= 255.0
            # NCHW模型做HWC转CHW；NHWC模型保持HWC
            if self.is_nchw:
                image_tensor = np.transpose(image_tensor, (2, 0, 1))
            # 连续内存化：保证送入推理的张量与模型输入布局完全一致
            image_tensor = np.ascontiguousarray(image_tensor)
            batch_tensors.append(image_tensor)
            image_shapes.append((h, w))
        return np.ascontiguousarray(np.stack(batch_tensors, axis=0)), image_shapes

    def multioutputs_decode(self,outputs):
        """
        这是3个检测层输出解码为统一候选数组的函数（多输出模型解码：
        逐检测层解码，cx=(x*2-0.5+grid_x)*stride，cy=(y*2-0.5+grid_y)*stride，
        w=(w*2)^2*anchor_w，h=(h*2)^2*anchor_h，再按层序合并）
        Args:
            outputs: 各检测层输出数组列表，每项shape为(1,C,ny,nx)，
                C=num_anchors*(5+num_classes)（通道布局为逐anchor的
                [x,y,w,h,obj,cls...]，即检测头卷积输出布局）
        Returns:
            统一候选数组：[x1,y1,x2,y2,score,cls_id]（模型输入尺度，
            已做两阶段低分过滤：先按obj过滤、再按类别最大分过滤）
        """
        per_anchor_c = 5 + self.num_classes
        rows_list = []
        for idx, branch in enumerate(outputs):
            bs, c, ny, nx = np.shape(branch)
            stride = self.STRIDES[idx]
            layer_anchors = self.anchors[idx]              # (na,2)
            num_anchors = len(layer_anchors)
            if c != num_anchors * per_anchor_c:
                raise ValueError("检测层{0}输出通道数({1})与num_anchors({2})*"
                                 "(5+类别数{3})({4})不符"
                                 .format(idx, c, num_anchors, self.num_classes,
                                         num_anchors * per_anchor_c))
            per_anchor = branch.reshape(bs, num_anchors, per_anchor_c, -1)
            # with_sigmoid=False时对整层输出经sigmoid一次广播激活
            # （xywh与obj/cls整体激活，与ONNX导出时x[i]=x[i].sigmoid()一致）
            if not self.with_sigmoid:
                per_anchor = sigmoid(per_anchor)
            # 分出bbox做解码（(C,grids)布局免转置）：
            # cx=(x*2-0.5+grid_x)*stride，cy=(y*2-0.5+grid_y)*stride，
            # w=(w*2)^2*anchor_w，h=(h*2)^2*anchor_h
            box = per_anchor[:, :, :4]                     # (bs,na,4,grids)
            grid = make_grid(ny, nx)                       # (2,grids) 对应形式
            cx = (box[:, :, 0] * 2.0 - 0.5 + grid[0][None, None, :]) * stride
            cy = (box[:, :, 1] * 2.0 - 0.5 + grid[1][None, None, :]) * stride
            bw = (box[:, :, 2] * 2.0) ** 2 * layer_anchors[:, 0][None, :, None]
            bh = (box[:, :, 3] * 2.0) ** 2 * layer_anchors[:, 1][None, :, None]
            boxes = np.stack([cx, cy, bw, bh], axis=2)     # (bs,na,4,grids) cxcywh
            # bbox与分数字段concat后统一一次transpose：
            # (bs,na,5+nc,grids) -> (bs,na,grids,5+nc) -> (bs,na*grids,5+nc)
            rows = np.concatenate([boxes, per_anchor[:, :, 4:]], axis=2)
            rows = rows.transpose(0, 1, 3, 2).reshape(bs, -1, per_anchor_c)
            rows_list.append(rows)
        # 按层序合并各检测层解码行（返回逐图像融合数组列表）
        merged_rows = np.concatenate(rows_list, axis=1)
        return [merged_rows[b] for b in range(bs)]

    def decode(self,dets):
        """
        这是融合输出解码为统一候选数组的函数（两阶段低分过滤：
        先按obj过滤、再按类别最大分过滤，同一阈值；bbox格式转换
        （cxcywh->xyxy）放在mask过滤之后再做）
        Args:
            dets: 融合输出数组，shape为(框数,5+num_classes)，
                列布局[cx,cy,w,h,obj,cls...]
        Returns:
            统一候选数组：[x1,y1,x2,y2,score,cls_id]（模型输入尺度）
        """
        if dets.shape[1] != 5 + self.num_classes:
            raise ValueError("融合输出列数({0})与5+类别数({1})不符"
                             .format(dets.shape[1], 5 + self.num_classes))
        # with_sigmoid=False时对obj/cls分数补做sigmoid（修复：单输出融合
        # 路径此前忽略with_sigmoid，obj/cls仍为logit致过滤与打分错误；
        # 前4列bbox已是解码后的cxcywh像素坐标，不做激活）
        if not self.with_sigmoid:
            dets = np.asarray(dets, dtype=np.float32).copy()
            dets[:, 4:] = sigmoid(dets[:, 4:])
        obj_conf = dets[:, 4]
        cls_scores = dets[:, 5:5 + self.num_classes]
        boxes = dets[:, 0:4]
        # 第一阶段低分过滤：先按obj过滤，后续argmax/max仅作用于存活行
        keep_obj = obj_conf > self.confidence_threshold
        boxes = boxes[keep_obj]
        obj_conf = obj_conf[keep_obj]
        cls_scores = cls_scores[keep_obj]
        if len(boxes) == 0:
            return np.zeros((0, 6), dtype=np.float32)
        # 第二阶段低分过滤：在存活行上选取类别最大分及其对应id，
        # 再按类别最大分过滤（同一阈值）
        cls_id = np.argmax(cls_scores, axis=1)
        cls_max = np.max(cls_scores, axis=1)
        keep_mask = cls_max > self.confidence_threshold
        boxes = boxes[keep_mask]
        # bbox格式转换（cxcywh -> xyxy）放在mask过滤之后再做
        cx, cy, w, h = (boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3])
        boxes = np.stack([cx - w / 2.0, cy - h / 2.0,
                          cx + w / 2.0, cy + h / 2.0], axis=1)
        # 打分：类别最大分×obj置信度
        scores = cls_max[keep_mask] * obj_conf[keep_mask]
        # 组装统一候选数组：[x1,y1,x2,y2,score,cls_id]
        preds = np.concatenate([boxes,
                                scores.reshape((-1, 1)),
                                cls_id[keep_mask].reshape((-1, 1))], axis=1)
        return preds

    def inference(self,image_tensor):
        """
        这是模型推理函数（与ONNXRuntimeEngine保持一致：IO Binding路径，
        输入经OrtValue绑定（CPU零拷贝借用/CUDA每帧一次H2D），输出复用
        预分配OrtValue缓冲，消除逐帧分配开销；低版本ONNXRuntime
        （<1.8无OrtValue预分配API）走session.run兜底）
        Args:
            image_tensor: 输入张量（预处理后的float32数组，shape为
                (1,3,H,W)的NCHW或(1,H,W,3)的NHWC，与模型输入一致）
        Returns:
            输出数组列表（每路输出一个数组，顺序与模型输出一致）
        """
        if not self.use_io_binding:
            # 低版本ONNXRuntime（<1.8）：无OrtValue预分配API，直接session.run兜底
            results = self.session.run(self.output_names,
                                       {self.input_name: image_tensor})
            return [np.asarray(out, dtype=np.float32) for out in results]
        # 组装输入OrtValue：按模型声明的dtype精确转换，
        # ascontiguousarray保证OrtValue可借用numpy缓冲（非连续切片会抛错）；
        # CPU路径零拷贝借用，CUDA路径每帧一次H2D
        dtype = TENSOR_TYPE_TO_NP.get(self.input_type, np.float32)
        arr = np.ascontiguousarray(image_tensor, dtype=dtype)
        if self.is_cuda:
            input_ort_value = ort.OrtValue.ortvalue_from_numpy(
                arr, "cuda", self.gpu_id)
        else:
            input_ort_value = ort.OrtValue.ortvalue_from_numpy(arr)
        # 创建IO Binding并绑定输入与预分配输出（静态shape输出复用预分配
        # 缓冲，动态shape输出按设备分配）
        io_binding = self.session.io_binding()
        io_binding.bind_ortvalue_input(self.input_name, input_ort_value)
        for name in self.output_names:
            if name in self.outputs:
                io_binding.bind_ortvalue_output(name, self.outputs[name])
            elif self.is_cuda:
                io_binding.bind_output(name, "cuda", self.gpu_id)
            else:
                io_binding.bind_output(name)
        # 模型推理（IO Binding路径）
        self.session.run_with_iobinding(io_binding)
        # 收集输出：预分配输出从复用缓冲取数，动态输出按绑定顺序取数；
        # 输出统一为float32（后处理契约按float32计算，fp16输出在此转换）
        results = []
        for idx, name in enumerate(self.output_names):
            if name in self.outputs:
                out = self.outputs[name].numpy()
            else:
                out = io_binding.get_outputs()[idx].numpy()
            results.append(np.asarray(out, dtype=np.float32))
        return results

    def detect(self,image):
        """
        这是对图像做检测的函数（支持单张图像(H,W,C)或批量图像
        (batch,H,W,C)/图像列表，与项目DetectionModel.detect一致；
        类内完成批量预处理、推理、解码、NMS与坐标还原）
        Args:
            image: 图像，opencv格式（BGR）；单张(H,W,C)或批量(batch,H,W,C)
        Returns:
            单张图像时返回检测结果字典列表（每项含bbox/score/cls_id/cls_name，
            bbox统一为原始图像尺度的x1y1x2y2格式）；批量图像时返回
            逐图像的检测结果列表
        """
        # 获取图像个数（单张升维为batch=1的批量形态）
        if isinstance(image, (list, tuple)):
            image_num = len(image)
            images = np.array(image)
        elif isinstance(image, np.ndarray):
            shape = np.shape(image)
            if len(shape) == 3:
                image_num = 1
                images = np.expand_dims(image, 0)
            elif len(shape) == 4:
                image_num = shape[0]
                images = image
            else:
                raise ValueError("输入图像必须为3维(H,W,C)或4维(batch,H,W,C)数组，"
                                 "实际shape为{0}".format(shape))
        else:
            raise ValueError("输入图像必须为图像数组或图像列表，实际类型为{0}"
                             .format(type(image).__name__))
        # 图像预处理（批量：letterbox、BGR转RGB、归一化、HWC转CHW；
        # 图像数量与batchsize校验在preprocess内完成，与项目一致）
        image_tensor, image_shapes = self.preprocess(images)
        # 模型推理（与项目后处理管线一致：多输出模型先经multioutputs_decode
        # 融合为逐图像单输出等价数组；单输出模型按图像切片，再逐图像经decode）
        outputs = self.inference(image_tensor)
        if len(outputs) > 1:
            fused_list = self.multioutputs_decode(outputs)
        else:
            fused_list = [outputs[0][i] for i in range(image_num)]
        # 逐图像后处理：decode->坐标还原->NMS->编码
        detection_outputs = []
        for i in range(image_num):
            preds = self.decode(fused_list[i])
            if len(preds) == 0:
                detection_outputs.append([])
                continue
            preds = rescale_coords(preds, self.model_hw, image_shapes[i])
            preds = nms(preds, self.iou_threshold)
            # 检测结果编码
            outputs_i = []
            for pred in preds:
                cls_id = int(pred[5])
                outputs_i.append({
                    "bbox": [round(pred[0]), round(pred[1]), round(pred[2]), round(pred[3])],
                    "score": round(pred[4], 4),
                    "cls_id": cls_id,
                    "cls_name": self.class_names[cls_id]})
            detection_outputs.append(outputs_i)
        if image_num == 1:      # 仅有一张图像，则进行降维
            return detection_outputs[0]
        return detection_outputs


    def detect_video(self,video_path,video_result_path,interval=-1,print_detection_result=False):
        """
        这是检测视频的函数（与项目DetectionModel.detect_video一致）
        Args:
            video_path: 视频路径
            video_result_path: 检测结果视频路径
            interval: 视频抽帧间隔(秒)，默认为-1逐帧检测
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
        # 初始化检测结果视频（视频结果统一保存为mp4）
        vid_writer = cv2.VideoWriter(video_result_path, cv2.VideoWriter_fourcc(*'mp4v'),
                                     fps, (w, h))
        if not vid_writer.isOpened():
            vid_cap.release()
            raise IOError("无法创建检测结果视频{0}（编码器不支持该尺寸）".format(video_result_path))
        # 抽帧步长：interval秒采样一次；fps异常时逐帧
        # 抽帧步长下限钳到1（修复：interval*fps<1时int截断为0，
        # cnt%step除零崩溃，如interval=0.03配30fps视频）
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
                    outputs = self.detect(frame)
                    if print_detection_result:
                        self.print_detection_results(outputs)
                    if len(outputs) > 0:
                        detect_image = draw_detection_results(frame, outputs, self.colors)
                elif cnt % step == 0:   # 间隔interval秒检测
                    outputs = self.detect(frame)
                    if print_detection_result:
                        self.print_detection_results(outputs)
                    if len(outputs) > 0:
                        detect_image = draw_detection_results(frame, outputs, self.colors)
                vid_writer.write(detect_image)
        finally:
            vid_cap.release()
            vid_writer.release()

    def print_detection_results(self,outputs):
        """
        这是打印检测结果的函数
        Args:
            outputs: 检测结果字典列表（bbox统一为x1y1x2y2格式）
        Returns:
        """
        for output in outputs:
            print("检测到{0}, bbox: {1},{2},{3},{4},score:{5:.4f}".format(
                output["cls_name"], *output["bbox"], output["score"]))

    def release(self):
        """
        这是释放推理资源的函数（与ONNXRuntimeEngine保持一致：
        InferenceSession没有显式close/release API，其原生内存
        （模型权重、provider状态）由引用计数在引用归零时析构回收，
        释放方式为置空session引用并清空预分配输出OrtValue缓冲；
        释放后实例不可再推理）
        """
        self.outputs = {}
        self.session = None



def detect(model,source,result_dir,interval=-1,print_detection_result=False):
    """
    这是利用检测模型检测图像或者视频的函数（单线程版，与detect.py功能一致）
    Args:
        model: 检测模型实例
        source: 图像文件或者图像文件夹或者视频文件路径或者视频文件夹
        result_dir: 结果保存路径（图像结果归入其下image/子目录，
            视频结果归入其下video/子目录）
        interval: 视频抽帧间隔(秒)，默认为-1逐帧检测
        print_detection_result: 是否打印检测结果，默认为False
    Returns:
    """
    # 初始化结果路径
    result_dir = os.path.abspath(result_dir)
    image_result_dir = os.path.join(result_dir, "image")
    video_result_dir = os.path.join(result_dir, "video")

    video_paths = []
    video_result_paths = []
    image_paths = []
    image_result_paths = []
    if os.path.isfile(source):               # 单个文件
        _, filename = os.path.split(source)
        fname, ext = os.path.splitext(filename)
        if ext in IMG_FORMATS:
            image_paths.append(os.path.abspath(source))
            image_result_paths.append(os.path.join(image_result_dir, fname + ".jpg"))
        elif ext in VID_FORMATS:
            video_paths.append(os.path.abspath(source))
            video_result_paths.append(os.path.join(video_result_dir, fname + ".mp4"))
    else:                                       # 批量文件（递归遍历子目录）
        # 递归遍历：源目录下多层子目录中的图像/视频都参与检测；
        # 子目录相对路径并入结果文件名，避免不同子目录同名文件互相覆盖
        for dirpath, _, filenames in sorted(os.walk(source)):
            rel_dir = os.path.relpath(dirpath, source)
            rel_prefix = "" if rel_dir == "." else rel_dir.replace(os.sep, "_") + "_"
            for filename in sorted(filenames):
                fname, ext = os.path.splitext(filename)
                if ext in IMG_FORMATS:
                    image_paths.append(os.path.join(dirpath, filename))
                    image_result_paths.append(os.path.join(image_result_dir, rel_prefix + fname + ".jpg"))
                elif ext in VID_FORMATS:
                    video_paths.append(os.path.join(dirpath, filename))
                    video_result_paths.append(os.path.join(video_result_dir, rel_prefix + fname + ".mp4"))

    # 检测图像（结果目录按需创建）
    if len(image_paths) > 0:
        if not os.path.exists(image_result_dir):
            os.makedirs(image_result_dir)
        print("开始检测图像，共有{0}张图片需要检测".format(len(image_paths)))
        detect_imageset(model, image_paths, image_result_paths, print_detection_result)
        print("结束检测图像")

    # 检测视频
    if len(video_paths) > 0:
        if not os.path.exists(video_result_dir):
            os.makedirs(video_result_dir)
        print("开始检测视频，共有{0}个视频需要检测".format(len(video_paths)))
        detect_videoset(model, video_paths, video_result_paths,
                        interval, print_detection_result)
        print("结束检测视频")


def detect_imageset(model,image_paths,image_result_paths,print_detection_result=False):
    """
    这是检测图像集的函数（单线程版，与detect.py功能一致）
    Args:
        model: 检测模型实例
        image_paths: 图像路径数组
        image_result_paths: 检测结果图像路径数组
        print_detection_result: 是否打印检测结果，默认为False
    Returns:
    """
    # 遍历所有图片进行检测
    for i in range(len(image_paths)):
        detect_single_image(model, image_paths[i], image_result_paths[i],
                            print_detection_result)


def detect_single_image(model,image_path,image_result_path,print_detection_result=False):
    """
    这是检测单张图像的函数（与detect.py功能一致）
    Args:
        model: 检测模型实例
        image_path: 图像路径
        image_result_path: 检测结果图像路径
        print_detection_result: 是否打印检测结果，默认为False
    Returns:
    """
    # 检测图像并绘制检测结果
    image = cv2.imread(image_path)
    if image is None:
        raise IOError("图像读取失败：{0}".format(image_path))
    outputs = model.detect(image)
    if print_detection_result:
        model.print_detection_results(outputs)
    # 绘制检测结果并保存（图像结果统一保存为jpg）
    draw_image = draw_detection_results(image, outputs, model.colors)
    cv2.imwrite(image_result_path, draw_image)


def detect_videoset(model,video_paths,video_result_paths,interval=-1,print_detection_result=False):
    """
    这是检测视频集的函数（单线程版，与detect.py功能一致）
    Args:
        model: 检测模型实例
        video_paths: 视频路径数组
        video_result_paths: 检测结果视频路径数组
        interval: 视频抽帧间隔(秒)，默认为-1逐帧检测
        print_detection_result: 是否打印检测结果，默认为False
    Returns:
    """
    # 遍历所有视频进行检测
    for i in range(len(video_paths)):
        model.detect_video(video_paths[i], video_result_paths[i],
                           interval, print_detection_result)


def run_main():
    """
    这是主函数（参数解析、模型初始化、图像/视频检测与资源释放）
    """
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="YOLOv5 ONNXRuntime部署推理测试脚本")
    parser.add_argument("--model", type=str, required=True, help="ONNX模型文件路径")
    parser.add_argument("--source", type=str, required=True, help="图像路径或图像目录或视频路径或视频目录")
    parser.add_argument("--names", type=str, default=None,
                        help="类别名文件路径（每行一个类别名；缺省为单类别person）")
    parser.add_argument("--confidence_threshold", type=float, default=0.6,
                        help="置信度阈值")
    parser.add_argument("--iou_threshold", type=float, default=0.45,
                        help="NMS抑制阈值")
    parser.add_argument("--is_normalize", type=int, default=1, choices=[0, 1],
                        help="是否做除以255归一化（1缺省；归一化已融入ONNX时设0）")
    parser.add_argument("--gpu_id", type=int, default=0,
                        help="GPU设备号（CUDA可用时生效，默认为0）")
    parser.add_argument("--result_dir", type=str, default="./result/detect",
                        help="检测结果保存目录（图像/视频结果归入其下image/与video/子目录）")
    parser.add_argument("--interval", type=float, default=-1,
                        help="视频抽帧间隔(秒)，-1为逐帧检测")
    parser.add_argument("--print_detection_result", action="store_true",
                        help="打印检测结果")
    args = parser.parse_args()

    # 类别名初始化（缺省为单类别person）
    if args.names is not None:
        with open(args.names, 'r') as f:
            class_names = [line.strip() for line in f.readlines() if line.strip()]
    else:
        class_names = ["person"]

    # 初始化检测模型
    model = YOLOv5ONNXRuntime(args.model, class_names,
                          confidence_threshold=args.confidence_threshold,
                          iou_threshold=args.iou_threshold,
                          is_normalize=bool(args.is_normalize),
                          gpu_id=args.gpu_id)

    # 初始化相关路径（结果目录：result_dir/<模型文件名>/<时间戳>/image|video）
    time_str = datetime.now().strftime('%Y%m%d%H%M%S')
    _, model_name = os.path.split(os.path.abspath(args.model))
    result_dir = os.path.join(args.result_dir, model_name, time_str)
    source = os.path.abspath(args.source)

    # 检测图像/视频（单线程）
    detect(model, source, result_dir, interval=args.interval,
           print_detection_result=args.print_detection_result)

    # 释放模型资源
    model.release()


if __name__ == "__main__":
    run_main()
