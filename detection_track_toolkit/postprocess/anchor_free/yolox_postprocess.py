# -*- coding: utf-8 -*-
# @Time    : 2026/9/10 08:15
# @Author  : DaiPuWei
# @Email   : daipuwei@qq.com
# @File    : yolox_postprocess.py
# @Software: PyCharm


"""
    这是定义YOLOX风格后处理器的脚本。

    模块内容：
        1. YOLOXPostprocessor：YOLOX风格后处理类（解耦检测头：
           输出坐标布局经bbox_format参数选择（'xyxy'缺省或'cxcywh'）+
           两阶段低分过滤（先obj过滤再按类别最大分二次过滤）+
           打分=类别最大分×obj + keep_aspect_ratio尺度还原），
           yolox直接使用。
        2. yolox_postprocess(...)：后处理器工厂函数，注册进POSTPROCESSOR_REGISTRY
           （注册名'yolox_postprocess'），由postprocess.build.build_postprocessor
           按名称创建。
"""

import math

import numpy as np

from ..base_postprocessor import BasePostprocessor
from ..build import POSTPROCESSOR_REGISTRY
from ..postprocess_utils import make_grid
from ..postprocess_utils import sigmoid

class YOLOXPostprocessor(BasePostprocessor):
    """
    这是YOLOX风格后处理类（解耦检测头：bbox坐标布局xyxy/cxcywh可选 +
    两阶段低分过滤（先obj过滤再按类别最大分二次过滤）+ 打分=类别最大分×obj +
    keep_aspect_ratio尺度还原）
    """

    def __init__(self,confidence_threshold,iou_threshold,class_names,model_hw,
                 bbox_format="xyxy",output_layout="grid_first",
                 with_sigmoid=True,**kwargs):
        """
        这是YOLOX风格后处理器的初始化函数
        Args:
            confidence_threshold: 置信度阈值
            iou_threshold: iou阈值
            class_names: 目标分类名称数组
            model_hw: 模型输入尺寸(height,width)
            bbox_format: 模型输出bbox坐标布局，'xyxy'（缺省）或'cxcywh'
                （多输出模型在multioutputs_decode中重新赋值为cxcywh）
            output_layout: 模型输出形状布局，'grid_first'（缺省，
                bs×grid_num×output_dim）或'dim_first'
                （bs×output_dim×grid_num，导出时输出前带transpose节点；
                多输出模型在multioutputs_decode中重新赋值为grid_first）
            with_sigmoid: 多输出原始模型ONNX中obj/cls是否已加sigmoid，
                True（缺省，ONNX已带sigmoid，multioutputs_decode不再激活）或
                False（ONNX未加sigmoid，经sigmoid封装函数手动向量化sigmoid）；
                bbox不做sigmoid
            kwargs: 其余参数透传给基类（nms_mode/class_agnostic_threshold/iou_mode）
        """
        if bbox_format not in ("xyxy", "cxcywh"):
            raise ValueError("未知的bbox_format'{0}'，支持xyxy或cxcywh"
                             .format(bbox_format))
        if output_layout not in ("grid_first", "dim_first"):
            raise ValueError("未知的output_layout'{0}'，支持grid_first或dim_first"
                             .format(output_layout))
        self.bbox_format = bbox_format
        self.output_layout = output_layout
        self.with_sigmoid = with_sigmoid
        super(YOLOXPostprocessor, self).__init__(confidence_threshold,iou_threshold,
                                                 class_names,model_hw,**kwargs)

    def decode(self,dets):
        """
        这是检测头原始输出解码为统一预测数组的函数（解耦检测头：
        输出列布局为num_classes+5（列4为obj_conf），bbox坐标布局
        x1y1x2y2/cxcywh经bbox_format参数选择；两阶段低分过滤：
        先按obj过滤，再在存活行上选取类别最大分做二次过滤；
        bbox格式转换放在mask过滤之后再做，置信度=类别最大分×obj_conf）
        Args:
            dets: 模型输出结果张量（num_classes+5输出，列4为obj_conf）
        Returns:
            预测数组：[x1,y1,x2,y2,score,cls_id]（bbox为模型输入尺度的xyxy）
        """
        dets = np.asarray(dets, dtype=np.float32)
        if dets.ndim == 1:
            dets = dets.reshape((1, -1))
        # 输出布局转换：dim_first（output_dim×grid_num，导出带transpose）
        # 转回grid_first（grid_num×output_dim）
        if self.output_layout == "dim_first":
            dets = np.transpose(dets, (1, 0))
        if dets.shape[1] != 5 + self.num_classes:
            raise ValueError("YOLOX输出列数({0})与5+类别数({1})不符"
                             .format(dets.shape[1], 5 + self.num_classes))
        obj_conf = dets[:, 4]
        cls_scores = dets[:, 5:5 + self.num_classes]
        boxes = dets[:, 0:4]
        # 第一阶段低分过滤：先按obj过滤，后续argmax/max仅作用于obj过滤后的存活行
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
        # bbox坐标布局转换（cxcywh -> xyxy，xyxy直通）放在mask过滤之后再做：
        # 仅对两阶段过滤后的存活行转换，减少转换行数
        if self.bbox_format == "cxcywh":
            cx, cy, w, h = (boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3])
            boxes = np.stack([cx - w / 2.0, cy - h / 2.0,
                              cx + w / 2.0, cy + h / 2.0], axis=1)
        # 打分：类别最大分×obj（yolox语义）
        scores = cls_max[keep_mask] * obj_conf[keep_mask]
        # 组装统一预测数组：[x1,y1,x2,y2,score,cls_id]
        preds = np.concatenate([boxes,
                               scores.reshape((-1, 1)),
                               cls_id[keep_mask].reshape((-1, 1))], axis=1)
        return preds

    def multioutputs_decode(self,branch_outputs):
        """
        这是多输出原始模型融合解码的函数（yolox家族；实现基类抽象方法，
        postprocess中自动调用）：各检测层输出先直接reshape为(bs,output_dim,grids)
        并按grid维concat，再分出bbox做解码（bbox不做sigmoid：中心点坐标=
        (x+grid)*stride数组，wh先指数化再乘stride数组），最后与分数字段concat后
        统一一次transpose，得到(bs,grid_num,5+nc)单输出等价数组
        Args:
            branch_outputs: 各检测层输出张量列表（引擎输出），每项shape为
                (batchsize,output_dim,h,w)或(batchsize,output_dim,h*w)，
                output_dim=5+num_classes（列布局[x,y,w,h,obj,cls...]）；
                以3输出模型为例（stride 8/16/32）：reshape后为
                (bs,output_dim,H*W/64)、(bs,output_dim,H*W/256)、(bs,output_dim,H*W/1024)
        Returns:
            每张图像的融合输出数组列表，每项shape为(框数,5+num_classes)，
            列布局[cx,cy,w,h,obj,cls...]（模型输入尺度像素；融合解码
            默认使用cxcywh格式，xyxy转换由decode按bbox_format统一完成）

        注意：各检测层stride无需指定，由model_hw与分支特征图hw自动推导
        （h=H/stride,w=W/stride；2D布局分支按s=sqrt(H*W/hw)推导）；
        grid以(2,grids)对应形式生成，解码全程(C,grids)布局免转置，
        最后concat后统一一次transpose。
        """
        # 多输出融合输出恒为(grid_num,5+nc)的grid_first布局、bbox恒为cxcywh格式：
        # 对output_layout与bbox_format重新赋值，保证后续decode按此格式处理
        # （融合解码默认使用cxcywh格式，xyxy转换由decode统一完成）
        self.output_layout = "grid_first"
        self.bbox_format = "cxcywh"
        batch = branch_outputs[0].shape[0]
        # 1) 各检测层直接reshape为(bs,output_dim,grids)，再按grid维concat
        flat_list = []
        grid_list = []
        stride_list = []
        for idx, branch in enumerate(branch_outputs):
            branch = np.asarray(branch, dtype=np.float32)
            if branch.ndim == 4:
                bs, c, h, w = branch.shape
                flat = branch.reshape(bs, c, h * w)
            else:
                bs, c, hw = branch.shape
                # stride由model_hw与特征图网格数hw自动推导：
                # h=H/s,w=W/s => h*w=H*W/s^2 => s=sqrt(H*W/hw)
                grid_total = self.model_hw[0] * self.model_hw[1]
                if grid_total % hw != 0:
                    raise ValueError("检测层{0}网格数({1})与模型输入尺寸{2}不符，"
                                     "无法推导stride".format(idx, hw, self.model_hw))
                stride = math.isqrt(grid_total // hw)
                if stride * stride != grid_total // hw:
                    raise ValueError("检测层{0}网格数({1})与模型输入尺寸{2}不符，"
                                     "无法推导整数stride".format(idx, hw, self.model_hw))
                h = self.model_hw[0] // stride
                w = self.model_hw[1] // stride
                flat = branch
            # stride由model_hw与特征图hw计算（H/h与W/w必须一致）
            if self.model_hw[0] % h != 0 or self.model_hw[1] % w != 0 or \
                    self.model_hw[0] // h != self.model_hw[1] // w:
                raise ValueError("检测层{0}特征图({1}x{2})与模型输入尺寸{3}不符，"
                                 "无法推导一致stride".format(idx, h, w, self.model_hw))
            stride = self.model_hw[0] // h
            if c != 5 + self.num_classes:
                raise ValueError("检测层{0}输出通道数({1})与5+类别数{2}({3})不符"
                                 .format(idx, c, self.num_classes, 5 + self.num_classes))
            flat_list.append(flat)
            grid_list.append(make_grid(h, w))           # (2,grids) 对应形式，免转置
            stride_list.append(np.full(h * w, stride, dtype=np.float32))
        merged = np.concatenate(flat_list, axis=2)      # (bs,output_dim,total_grids)
        grid = np.concatenate(grid_list, axis=1)        # (2,total_grids) 网格单元坐标
        strides = np.concatenate(stride_list)           # (total_grids,) 逐网格stride数组
        # 2) 分出bbox做解码（bbox不做sigmoid，全程(C,grids)布局免转置）：
        #    中心点坐标=(x+grid)*stride数组，wh先指数化再乘stride数组；
        #    解码默认使用cxcywh格式，xyxy转换由decode按bbox_format统一完成
        raw_xy = merged[:, :2]
        raw_wh = merged[:, 2:4]
        score_part = merged[:, 4:]
        xy = (raw_xy + grid) * strides
        wh = np.exp(raw_wh) * strides
        boxes = np.concatenate([xy, wh], axis=1)
        # 3) ONNX中obj/cls未加sigmoid时经sigmoid封装函数手动完成向量化sigmoid
        if not self.with_sigmoid:
            score_part = sigmoid(score_part)
        # 4) bbox与分数字段concat后统一一次transpose：
        #    (bs,grid_num,5+nc) -> 逐图像数组列表
        rows = np.concatenate([boxes, score_part], axis=1).transpose(0, 2, 1)
        return [rows[b] for b in range(batch)]

    def encode(self,preds):
        """
        这是检测结果编码的函数（bbox/score/cls_id/cls_name）
        Args:
            preds: 保留框的预测数组（[x1,y1,x2,y2,score,cls_id]）
        Returns:
            检测结果字典列表，每项含bbox/score/cls_id/cls_name
        """
        outputs = []
        for pred in preds:
            cls_id = int(pred[5])
            outputs.append({"bbox": [round(pred[0]), round(pred[1]), round(pred[2]), round(pred[3])],
                            "score": round(pred[4], 4),
                            "cls_id": cls_id,
                            "cls_name": self.class_names[cls_id]})
        return outputs


@POSTPROCESSOR_REGISTRY.register()
def yolox_postprocess(confidence_threshold,iou_threshold,class_names,model_hw,**kwargs):
    """
    这是初始化YOLOX风格后处理器的工厂函数
    Args:
        confidence_threshold: 置信度阈值
        iou_threshold: iou阈值
        class_names: 目标分类名称数组
        model_hw: 模型输入尺寸(height,width)
        kwargs: 自定义参数字典（透传给后处理器类）
    Returns:
        YOLOXPostprocessor实例
    """
    return YOLOXPostprocessor(confidence_threshold,iou_threshold,
                              class_names,model_hw,**kwargs)
