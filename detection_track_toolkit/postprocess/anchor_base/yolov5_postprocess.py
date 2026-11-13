# -*- coding: utf-8 -*-
# @Time    : 2026/9/10 08:16
# @Author  : DaiPuWei
# @Email   : daipuwei@qq.com
# @File    : yolov5_postprocess.py
# @Software: PyCharm

"""
    这是定义YOLOv5风格后处理器的脚本。

    模块内容：
        1. YOLOv5Postprocessor：YOLOv5风格后处理类（anchor base家族解码头：
           融合输出为(num_boxes,5+num_classes)，列布局[cx,cy,w,h,obj,cls...]，
           解码为xyxy坐标 + obj×cls打分 + keep_aspect_ratio尺度还原），
           yolov5/yolov7直接使用；
        2. yolov5_postprocess(...)：后处理器工厂函数，注册进POSTPROCESSOR_REGISTRY
           （注册名'yolov5_postprocess'），由postprocess.build.build_postprocessor
           按名称创建。
"""

import math

import numpy as np

from ..base_postprocessor import BasePostprocessor
from ..build import POSTPROCESSOR_REGISTRY
from ..postprocess_utils import make_grid
from ..postprocess_utils import sigmoid

class YOLOv5Postprocessor(BasePostprocessor):
    """
    这是YOLOv5风格后处理类（anchor base解码头：cxcywh+obj置信度，
    解码为xyxy + obj×cls打分 + keep_aspect_ratio尺度还原）
    """

    def __init__(self,confidence_threshold,iou_threshold,class_names,model_hw,
                 with_sigmoid=True,anchors=None,**kwargs):
        """
        这是YOLOv5风格后处理器的初始化函数
        Args:
            confidence_threshold: 置信度阈值
            iou_threshold: iou阈值
            class_names: 目标分类名称数组
            model_hw: 模型输入尺寸(height,width)
            with_sigmoid: 多输出原始模型ONNX中检测层输出（xywh与obj/cls）是否已加
                sigmoid，True（缺省，ONNX已带sigmoid，解码不再激活）或False
                （ONNX未加sigmoid，multioutputs_decode经sigmoid整层广播激活）
            anchors: 各检测层anchor宽高列表，shape为(层数,每层anchor数,2)，
                yolov5多输出原始模型解码必需（训练侧anchor，无法由模型推导；
                经anchors_path读取，每行一个数值，缺省为coco_anchors.txt，
                层数由engine输出节点个数决定）；单输出模型忽略。各检测层
                stride无需指定，由model_hw与分支特征图hw自动推导
            kwargs: 其余参数透传给基类（nms_mode/class_agnostic_threshold/iou_mode）
        """
        self.with_sigmoid = with_sigmoid
        self.anchors = anchors
        super(YOLOv5Postprocessor, self).__init__(confidence_threshold,iou_threshold,
                                                   class_names,model_hw,**kwargs)

    def decode(self,dets):
        """
        这是检测头原始输出解码为统一预测数组的函数（anchor base家族：
        融合输出列布局为[cx,cy,w,h,obj,cls...]，两阶段低分过滤：
        先按obj过滤，再在存活行上选取类别最大分做二次过滤；
        bbox格式转换（cxcywh->xyxy）放在mask过滤之后再做，
        置信度=类别最大分×obj）
        Args:
            dets: 模型输出结果张量，shape为(框数,5+num_classes)
        Returns:
            预测数组：[x1,y1,x2,y2,score,cls_id]（bbox为模型输入尺度的xyxy）
        """
        dets = np.asarray(dets, dtype=np.float32)
        if dets.ndim == 1:
            dets = dets.reshape((1, -1))
        num_classes = self.num_classes
        if dets.shape[1] != 5 + num_classes:
            raise ValueError("YOLOv5融合输出列数({0})与5+类别数({1})不符"
                             .format(dets.shape[1], 5 + num_classes))
        obj_conf = dets[:, 4]
        cls_scores = dets[:, 5:5 + num_classes]
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
        # bbox格式转换（cxcywh -> xyxy）放在mask过滤之后再做：
        # 仅对两阶段过滤后的存活行转换，减少转换行数
        cx, cy, w, h = (boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3])
        boxes = np.stack([cx - w / 2.0, cy - h / 2.0,
                          cx + w / 2.0, cy + h / 2.0], axis=1)
        # 打分：类别最大分×obj置信度（NMS前削减大量低分候选）
        scores = cls_max[keep_mask] * obj_conf[keep_mask]
        # 组装统一预测数组：[x1,y1,x2,y2,score,cls_id]
        preds = np.concatenate([boxes,
                                scores.reshape((-1, 1)),
                                cls_id[keep_mask].reshape((-1, 1))], axis=1)
        return preds

    def multioutputs_decode(self,branch_outputs):
        """
        这是多输出模型bbox解码的函数，解码流程：

        逐检测层解码：
            cx=(x*2-0.5+grid_x)*stride，
            cy=(y*2-0.5+grid_y)*stride，
            w=(w*2)^2*anchor_w，
            h=(h*2)^2*anchor_h，
        再按层序合并为(bs,na*total_grids,5+nc)单输出等价数组
        Args:
            branch_outputs: 各检测层输出张量列表（引擎输出），每项shape为
                (batchsize,C,ny,nx)或(batchsize,C,ny*nx)，
                C=num_anchors*(5+num_classes)（通道布局为逐anchor的
                [x,y,w,h,obj,cls...]，即检测头卷积输出布局）
        Returns:
            每张图像的融合输出数组列表，每项shape为(框数,5+num_classes)，
            列布局[cx,cy,w,h,obj,cls...]（模型输入尺度像素，与单输出融合
            格式一致；行序为层序：层0各anchor行在前，
            其余检测层依次在后）

        注意：各检测层stride无需指定，由model_hw与分支特征图hw自动推导
        （ny=H/stride,nx=W/stride；2D布局分支按s=sqrt(H*W/hw)推导）；
        anchors经anchors_path配置（训练侧anchor，无法由模型推导）；
        grid以(2,grids)对应形式生成，解码全程(C,grids)布局免转置，
        逐层解码后统一一次transpose并按层序concat。
        """
        if self.anchors is None:
            raise ValueError("多输出yolov5模型解码需要anchors参数（经anchors_path配置）")
        if len(self.anchors) != len(branch_outputs):
            raise ValueError("anchors层数({0})与多输出分支数({1})不一致"
                             .format(len(self.anchors), len(branch_outputs)))
        batch = branch_outputs[0].shape[0]
        per_anchor_c = 5 + self.num_classes
        rows_list = []
        for idx, branch in enumerate(branch_outputs):
            branch = np.asarray(branch, dtype=np.float32)
            # 1) 各检测层直接reshape为(bs,C,grids)
            if branch.ndim == 4:
                bs, c, ny, nx = branch.shape
                flat = branch.reshape(bs, c, ny * nx)
            else:
                bs, c, hw = branch.shape
                # stride由model_hw与特征图网格数hw自动推导：
                # ny=H/s,nx=W/s => ny*nx=H*W/s^2 => s=sqrt(H*W/hw)
                grid_total = self.model_hw[0] * self.model_hw[1]
                if grid_total % hw != 0:
                    raise ValueError("检测层{0}网格数({1})与模型输入尺寸{2}不符，"
                                     "无法推导stride".format(idx, hw, self.model_hw))
                stride = math.isqrt(grid_total // hw)
                if stride * stride != grid_total // hw:
                    raise ValueError("检测层{0}网格数({1})与模型输入尺寸{2}不符，"
                                     "无法推导整数stride".format(idx, hw, self.model_hw))
                ny = self.model_hw[0] // stride
                nx = self.model_hw[1] // stride
                flat = branch
            # stride由model_hw与特征图hw计算（H/ny与W/nx必须一致）
            if self.model_hw[0] % ny != 0 or self.model_hw[1] % nx != 0 or \
                    self.model_hw[0] // ny != self.model_hw[1] // nx:
                raise ValueError("检测层{0}特征图({1}x{2})与模型输入尺寸{3}不符，"
                                 "无法推导一致stride".format(idx, ny, nx, self.model_hw))
            stride = self.model_hw[0] // ny
            layer_anchors = np.asarray(self.anchors[idx], dtype=np.float32)  # (na,2)
            num_anchors = layer_anchors.shape[0]
            if c != num_anchors * per_anchor_c:
                raise ValueError("检测层{0}输出通道数({1})与num_anchors({2})*"
                                 "(5+类别数{3})({4})不符"
                                 .format(idx, c, num_anchors, self.num_classes,
                                         num_anchors * per_anchor_c))
            # 2) sigmoid门控（多输出导出已含sigmoid时with_sigmoid=True不再激活；
            #    未含时经sigmoid封装函数对整层输出一次广播完成，
            #    xywh与obj/cls整体激活）
            if not self.with_sigmoid:
                flat = sigmoid(flat)
            # 3) 分出bbox做解码（全程(C,grids)布局免转置）：
            #    cx=(x*2-0.5+grid_x)*stride，cy=(y*2-0.5+grid_y)*stride，
            #    w=(w*2)^2*anchor_w，h=(h*2)^2*anchor_h
            per_anchor = flat.reshape(batch, num_anchors, per_anchor_c, -1)
            box = per_anchor[:, :, :4]                  # (bs,na,4,grids)
            grid = make_grid(ny, nx)                    # (2,grids) 对应形式，免转置
            cx = (box[:, :, 0] * 2.0 - 0.5 + grid[0][None, None, :]) * stride
            cy = (box[:, :, 1] * 2.0 - 0.5 + grid[1][None, None, :]) * stride
            bw = (box[:, :, 2] * 2.0) ** 2 * layer_anchors[:, 0][None, :, None]
            bh = (box[:, :, 3] * 2.0) ** 2 * layer_anchors[:, 1][None, :, None]
            boxes = np.stack([cx, cy, bw, bh], axis=2)  # (bs,na,4,grids) cxcywh
            # 4) bbox与分数字段concat后统一一次transpose：
            #    (bs,na,5+nc,grids) -> (bs,na,grids,5+nc) -> (bs,na*grids,5+nc)
            rows = np.concatenate([boxes, per_anchor[:, :, 4:]], axis=2)
            rows = rows.transpose(0, 1, 3, 2).reshape(batch, -1, per_anchor_c)
            rows_list.append(rows)
        # 5) 按层序合并各检测层解码行
        merged_rows = np.concatenate(rows_list, axis=1)
        return [merged_rows[b] for b in range(batch)]

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
def yolov5_postprocess(confidence_threshold,iou_threshold,class_names,model_hw,**kwargs):
    """
    这是初始化YOLOv5风格后处理器的工厂函数
    Args:
        confidence_threshold: 置信度阈值
        iou_threshold: iou阈值
        class_names: 目标分类名称数组
        model_hw: 模型输入尺寸(height,width)
        kwargs: 自定义参数字典（透传给后处理器类）
    Returns:
        YOLOv5Postprocessor实例
    """
    return YOLOv5Postprocessor(confidence_threshold,iou_threshold,
                               class_names,model_hw,**kwargs)
