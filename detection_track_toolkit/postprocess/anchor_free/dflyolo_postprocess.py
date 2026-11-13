# -*- coding: utf-8 -*-
# @Time    : 2026/9/10 08:15
# @Author  : DaiPuWei
# @Email   : daipuwei@qq.com
# @File    : dflyolo_postprocess.py
# @Software: PyCharm

"""
    这是定义DFL检测头YOLO后处理器的脚本。

    模块内容：
        1. DFLYOLOPostprocessor：DFL检测头YOLO后处理类（anchor free DFL家族）。
           单输出ONNX的输出格式确认为4+1+num_classes：
           - 4为bbox四维坐标，可能是x1y1x2y2（缺省）也可能是cxcywh
             （bbox_format参数选择）；
           - 1为scores_sum分量（第4列，紧跟bbox之后），可要可不要
             （按列数自动识别：4+num_classes列无scores_sum，
             5+num_classes列含scores_sum），存在时严格执行先按scores_sum
             低分过滤、再在存活行上选取类别最大分做二次过滤；
           打分=类别最大分+keep_aspect_ratio尺度还原，yolov6/v8/v9/v10/
           v11/v12/v13/v26/ppyoloe直接使用；
        2. dflyolo_postprocess(...)：后处理器工厂函数，注册进POSTPROCESSOR_REGISTRY
           （注册名'dflyolo_postprocess'），由postprocess.build.build_postprocessor
           按名称创建。
"""

import math

import numpy as np

from ..base_postprocessor import BasePostprocessor
from ..build import POSTPROCESSOR_REGISTRY
from ..postprocess_utils import dfl_decode
from ..postprocess_utils import dist2bbox
from ..postprocess_utils import make_grid
from ..postprocess_utils import sigmoid


class DFLYOLOPostprocessor(BasePostprocessor):
    """
    这是DFL检测头YOLO后处理类（anchor free DFL家族：单输出ONNX格式
    确认为4+1+num_classes，scores_sum可缺省，置信度=类别最大分）
    """

    def __init__(self,confidence_threshold,iou_threshold,class_names,model_hw,
                 bbox_format="xyxy",output_layout="grid_first",**kwargs):
        """
        这是DFL检测头YOLO后处理器的初始化函数
        Args:
            confidence_threshold: 置信度阈值
            iou_threshold: iou阈值
            class_names: 目标分类名称数组
            model_hw: 模型输入尺寸(height,width)
            bbox_format: 模型输出bbox坐标布局，'xyxy'（缺省）或'cxcywh'
                （多输出模型在multioutputs_decode中重新赋值为xyxy）
            output_layout: 模型输出形状布局，'grid_first'（缺省，
                bs×grid_num×output_dim）或'dim_first'
                （bs×output_dim×grid_num，导出时输出前带transpose节点；
                多输出模型在multioutputs_decode中重新赋值为grid_first）
            kwargs: 其余参数透传给基类（nms_mode/class_agnostic_threshold/iou_mode）

        reg_max与with_score_sum在多输出解码（multioutputs_decode）时由特征图
        通道数与类别数自动判断更新（self.reg_max/self.with_score_sum），
        单输出模型用不到。
        """
        if bbox_format not in ("xyxy", "cxcywh"):
            raise ValueError("未知的bbox_format'{0}'，支持xyxy或cxcywh"
                             .format(bbox_format))
        if output_layout not in ("grid_first", "dim_first"):
            raise ValueError("未知的output_layout'{0}'，支持grid_first或dim_first"
                             .format(output_layout))
        self.bbox_format = bbox_format
        self.output_layout = output_layout
        # DFL回归区间数与scores_sum存在性：多输出解码时由特征图通道数
        # 与类别数自动判断更新（见multioutputs_decode）
        self.reg_max = None
        self.with_score_sum = None
        super(DFLYOLOPostprocessor, self).__init__(confidence_threshold,iou_threshold,
                                                   class_names,model_hw,**kwargs)

    def decode(self,dets):
        """
        这是检测头原始输出解码为统一预测数组的函数（anchor free DFL家族，
        自适应scores_sum有无（4/5+nc）与输出布局（grid_first/dim_first）
        两种导出形态，详见模块docstring；两阶段低分过滤（先scores_sum
        后类别最大分），bbox格式转换放在mask过滤之后再做）
        Args:
            dets: 模型输出结果张量
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
        num_classes = self.num_classes
        n_cols = dets.shape[1]
        # 单输出ONNX格式确认：4+1+num_classes
        # （4=bbox四维坐标；1=scores_sum第4列，可要可不要，按列数自动识别：
        #  4+num_classes列无scores_sum，5+num_classes列含scores_sum）
        if n_cols != 4 + num_classes and n_cols != 5 + num_classes:
            raise ValueError("DFL输出列数({0})与4/5+类别数({1})不符"
                             .format(n_cols, num_classes))
        boxes = dets[:, 0:4]
        if n_cols == 5 + num_classes:
            # scores_sum在第4列（紧跟bbox之后），仅用于低分预过滤
            scores_sum = dets[:, 4]
            cls_scores = dets[:, 5:5 + num_classes]
        else:
            cls_scores = dets[:, 4:4 + num_classes]
            scores_sum = None
        # 第一阶段低分过滤：scores_sum存在时严格执行——先按scores_sum过滤，
        # 后续argmax/max仅作用于scores_sum过滤后的存活行
        if scores_sum is not None:
            keep_sum = scores_sum > self.confidence_threshold
            boxes = boxes[keep_sum]
            cls_scores = cls_scores[keep_sum]
            if len(boxes) == 0:
                return np.zeros((0, 6), dtype=np.float32)
        # 第二阶段低分过滤：在存活行上选取类别最大分及其对应id，
        # 再按类别最大分过滤（同一阈值）
        cls_id = np.argmax(cls_scores, axis=1)
        score = np.max(cls_scores, axis=1)
        keep_mask = score > self.confidence_threshold
        boxes = boxes[keep_mask]
        # 坐标布局转换（cxcywh -> xyxy，xyxy直通）放在mask过滤之后再做：
        # 仅对两阶段过滤后的存活行转换，减少转换行数
        if self.bbox_format == "cxcywh":
            cx, cy, w, h = (boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3])
            boxes = np.stack([cx - w / 2.0, cy - h / 2.0,
                              cx + w / 2.0, cy + h / 2.0], axis=1)
        # 组装统一预测数组：[x1,y1,x2,y2,score,cls_id]
        preds = np.concatenate([boxes,
                                score[keep_mask].reshape((-1, 1)),
                                cls_id[keep_mask].reshape((-1, 1))], axis=1)
        return preds

    def multioutputs_decode(self,branch_outputs):
        """
        这是多输出原始模型融合解码的函数（DFL家族；实现基类抽象方法，
        postprocess中自动调用）：各检测层输出先直接reshape为
        (bs,output_dim,grids)并按grid维concat，再分出bbox做DFL解码
        （经dfl_decode做softmax加权（reg_max=1时仅reshape）、dist2bbox
        转xyxy像素坐标两个封装函数完成），分数字段按with_score_sum处理（True说明ONNX本身有sigmoid
        节点直接取用；False则手动sigmoid并补算scores_sum），最后与bbox
        concat后统一一次transpose，得到(bs,grid_num,5+nc)单输出等价数组
        Args:
            branch_outputs: 各检测层输出张量列表（引擎输出），每项shape为
                (batchsize,output_dim,h,w)或(batchsize,output_dim,h*w)，
                output_dim=4*reg_max+num_classes（无scores_sum）或
                4*reg_max+1+num_classes（含scores_sum，按输出通道数自动识别）
        Returns:
            每张图像的融合输出数组列表，每项shape为(框数,5+num_classes)，
            列布局[x1,y1,x2,y2,scores_sum,cls...]（模型输入尺度像素；
            融合解码使用xyxy格式，scores_sum缺省时已按
            clip(sum(sigmoid(cls)),0,1)补算）

        注意：各检测层stride无需指定，由model_hw与分支特征图hw自动推导
        （h=H/stride,w=W/stride；2D布局分支按s=sqrt(H*W/hw)推导）；
        reg_max与scores_sum存在性由特征图通道数与类别数自动判断更新
        （self.reg_max/self.with_score_sum）；with_score_sum为True说明ONNX
        本身有sigmoid节点（分数已激活），为False则手动sigmoid并补算
        scores_sum；grid以(2,grids)对应形式生成，解码全程(C,grids)布局
        免转置，最后concat后统一一次transpose。
        """
        # 多输出融合输出恒为(grid_num,5+nc)的grid_first布局、bbox恒为xyxy格式：
        # 对output_layout与bbox_format重新赋值，保证后续decode按此格式处理
        self.output_layout = "grid_first"
        self.bbox_format = "xyxy"
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
            # reg_max与scores_sum存在性由特征图通道数c与类别数自动判断：
            # 4*reg_max+nc（无scores_sum）或4*reg_max+1+nc（含scores_sum）
            if c - self.num_classes >= 4 and (c - self.num_classes) % 4 == 0:
                reg_max, with_score_sum = (c - self.num_classes) // 4, False
            elif c - 1 - self.num_classes >= 4 and (c - 1 - self.num_classes) % 4 == 0:
                reg_max, with_score_sum = (c - 1 - self.num_classes) // 4, True
            else:
                raise ValueError("检测层{0}输出通道数({1})与4*reg_max(±scores_sum)+"
                                 "类别数{2}不符，无法推导reg_max"
                                 .format(idx, c, self.num_classes))
            if idx == 0:
                # 首层推导结果更新至实例（self.reg_max/self.with_score_sum）
                self.reg_max = reg_max
                self.with_score_sum = with_score_sum
            elif reg_max != self.reg_max or with_score_sum != self.with_score_sum:
                raise ValueError("检测层{0}推导的reg_max({1})/scores_sum({2})与层0"
                                 "({3}/{4})不一致"
                                 .format(idx, reg_max, with_score_sum,
                                         self.reg_max, self.with_score_sum))
            flat_list.append(flat)
            grid_list.append(make_grid(h, w, offset=0.5))  # (2,grids) 网格中心点
            stride_list.append(np.full(h * w, stride, dtype=np.float32))
        merged = np.concatenate(flat_list, axis=2)      # (bs,output_dim,total_grids)
        grid_center = np.concatenate(grid_list, axis=1)  # (2,total_grids) 网格中心点坐标
        strides = np.concatenate(stride_list)           # (total_grids,) 逐网格stride数组
        # 2) 分出bbox做DFL解码（bbox不做sigmoid，全程(C,grids)布局免转置；
        #    dfl_decode做softmax加权得偏移量（reg_max=1时无分布区间仅reshape，
        #    通道即[l,t,r,b]四个偏移量），再经dist2bbox得xyxy像素坐标）
        n_box = 4 * self.reg_max
        box = merged[:, :n_box].reshape(batch, 4, self.reg_max, -1)
        offsets = dfl_decode(box, self.reg_max)
        boxes = dist2bbox(offsets, grid_center, bbox_format=self.bbox_format) * strides
        # 3) 分数字段：with_score_sum为True说明ONNX本身有sigmoid节点
        #    （首通道为scores_sum、其后为cls，分数已激活无需处理）；
        #    为False则分数字段为纯cls未激活——先经sigmoid封装函数手动完成
        #    向量化sigmoid，再按scores_sum=clip(sum(sigmoid(cls)),0,1)
        #    补算scores_sum并拼接到首通道（与含scores_sum的ONNX输出形态一致）
        score_part = merged[:, n_box:]
        if not self.with_score_sum:
            score_part = sigmoid(score_part)
            scores_sum = np.clip(score_part.sum(axis=1, keepdims=True), 0, 1)
            score_part = np.concatenate([scores_sum, score_part], axis=1)
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
def dflyolo_postprocess(confidence_threshold,iou_threshold,class_names,model_hw,**kwargs):
    """
    这是初始化DFL检测头YOLO后处理器的工厂函数
    Args:
        confidence_threshold: 置信度阈值
        iou_threshold: iou阈值
        class_names: 目标分类名称数组
        model_hw: 模型输入尺寸(height,width)
        kwargs: 自定义参数字典（bbox_format/nms_mode等，透传给后处理器类）
    Returns:
        DFLYOLOPostprocessor实例
    """
    return DFLYOLOPostprocessor(confidence_threshold,iou_threshold,
                                class_names,model_hw,**kwargs)
