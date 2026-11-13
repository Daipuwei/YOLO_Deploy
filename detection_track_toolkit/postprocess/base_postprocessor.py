# -*- coding: utf-8 -*-
# @Time    : 2026/9/10 05:26
# @Author  : DaiPuWei
# @Email   : daipuwei@qq.com
# @File    : base_postprocessor.py
# @Software: PyCharm

"""
    这是定义检测模型后处理器抽象基类的脚本。

    模块内容：
        1. BasePostprocessor：后处理器抽象基类（ABC），固定检测结果解码流水线
           （解码→尺度还原→按类别NMS→结果编码），全流水线以预测数组preds为唯一
           中间表示：preds = [x1,y1,x2,y2,score,cls_id,附加特征列...]
           （bbox为模型输入尺度的xyxy；关键点/投影点等附加特征列作为"乘客"
           原样透传、跟随行下标存活）：
           - decode(dets)为抽象方法，子类必须实现：检测头原始输出→preds
             （bbox解码+打分+预过滤，检测头差异所在）；
           - rescale_coords(preds, model_hw, image_shape, extra_coord_pairs)：
             统一尺度还原独立函数（流水线直接调用），bbox列(0:4)始终还原，
             extra_coord_pairs声明的附加坐标列对做相同还原（缺省为空元组，
             关键点等模型仅需配置无需覆写）；
           - nms(preds, score_threshold, iou_threshold, mode,
             class_agnostic_threshold, iou_mode)：统一NMS独立函数
             （流水线直接调用），过滤模式由实例配置nms_mode决定：
             class_agnostic（缺省）整体过滤：所有类别混在一起做贪心NMS，
             与原始cv2语义一致；per_class两阶段：先按类别过滤（不同类别互不抑制，
             修复类别无关NMS漏检），再对所有保留框做一次整体过滤
             （第二阶段必须执行，阈值缺省与iou_threshold相同，可经
             class_agnostic_threshold单独指定，如0.9）；
             抑制度量类型由实例配置iou_mode决定（iou/giou/diou，见
             postprocess.postprocess_utils.iou_xyxy），纯numpy自实现按xyxy计算几何
             （修复cv2.dnn.NMSBoxes按[x,y,w,h]解释矩形导致IoU几何算错的问题）；
           - encode(preds)：检测结果编码抽象函数，子类必须实现（输出格式差异所在），
             各风格类各自实现（bbox/score/cls_id/cls_name），
             关键点等模型覆写加附加字段；
           - 空输出（0个候选框）直接返回空结果（修复空输出argmax崩溃）；
           - multioutputs_decode(branch_outputs)为抽象方法，子类必须实现
             （多输出原始模型解码，家族解码逻辑差异所在）；
           - postprocess_single(dets, image_shape)对单张图像的检测头输出做完整后处理；
           - postprocess(outputs, image_shapes)为总体后处理入口，对批量图像的检测头
             输出做完整后处理（多输出原始模型先经multioutputs_decode融合为单输出
             等价数组，再逐张调用postprocess_single，无检测目标返回[]）。

        子类：postprocess.yolov5.yolov5_postprocess.YOLOv5Postprocessor（YOLOv5风格）、
              postprocess.yolox.yolox_postprocess.YOLOXPostprocessor（YOLOX风格）、
              postprocess.dflyolo.dflyolo_postprocess.DFLYOLOPostprocessor（DFLYOLO风格），
        均由 postprocess.build.POSTPROCESSOR_REGISTRY 注册工厂函数，
        由 build_postprocessor 按名称创建。
        核心算法沉淀为无状态独立函数（postprocess.postprocess_utils.iou_xyxy/nms/
        rescale_coords），可脱离类独立调用；流水线直接调用独立函数。
"""

import numpy as np

from abc import ABC, abstractmethod

from .postprocess_utils import nms
from .postprocess_utils import rescale_coords
from .postprocess_utils import validate_nms_mode
from .postprocess_utils import validate_iou_mode

class BasePostprocessor(ABC):

    def __init__(self,confidence_threshold,iou_threshold,class_names,model_hw,
                 nms_mode="class_agnostic",class_agnostic_threshold=None,
                 iou_mode="iou"):
        """
        这是检测模型后处理器抽象基类的初始化函数
        Args:
            confidence_threshold: 置信度阈值
            iou_threshold: iou阈值
            class_names: 目标分类名称数组
            model_hw: 模型输入尺寸(height,width)，用于检测框尺度还原
            nms_mode: NMS过滤模式，默认为'class_agnostic'（整体过滤，与原始
                cv2语义一致）；'per_class'为按类别过滤后整体过滤（两阶段，
                第二阶段整体过滤必须执行，缺省阈值取iou_threshold）
            class_agnostic_threshold: per_class模式第二阶段整体过滤阈值，
                缺省None即用iou_threshold做全局过滤（跨类别重复框按iou_threshold
                抑制）；大于0时按该阈值做全局过滤（如0.9只抑制高度重叠的
                跨类别重复框）；0或负显式禁用全局过滤（仅按类别过滤，
                跨类别重复框保留）；class_agnostic模式忽略
            iou_mode: NMS抑制度量类型，默认为'iou'，候选有giou/diou
                （度量定义见postprocess.postprocess_utils.iou_xyxy）
        """
        validate_nms_mode(nms_mode)
        validate_iou_mode(iou_mode)
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.class_names = class_names
        self.num_classes = len(self.class_names)
        self.model_hw = model_hw
        # 附加坐标列对声明（如关键点x,y列号），rescale时与bbox列做相同还原；
        # 关键点等模型在初始化后设置，如self.extra_coord_pairs = ((6,7),(8,9))
        self.extra_coord_pairs = ()
        # NMS过滤模式、总体NMS阈值与抑制度量类型
        self.nms_mode = nms_mode
        self.class_agnostic_threshold = class_agnostic_threshold
        self.iou_mode = iou_mode

    def release(self):
        """
        这是释放后处理器资源的函数（契约声明，缺省空实现：
        当前后处理器仅持有纯CPU配置，无原生资源；未来持有
        GPU缓冲/文件句柄等资源的子类应覆写本方法，
        模型侧release()会统一级联调用）
        """

    @abstractmethod
    def decode(self,dets):
        """
        这是检测头原始输出解码为统一预测数组的抽象函数（子类必须实现，检测头差异所在）
        Args:
            dets: 模型输出结果张量
        Returns:
            预测数组，shape为(框数,6+E)，每行为
            [x1,y1,x2,y2,score,cls_id,附加特征列...]（bbox为模型输入尺度的xyxy）
        """
        raise NotImplementedError

    @abstractmethod
    def multioutputs_decode(self,branch_outputs):
        """
        这是多输出原始模型融合解码的抽象函数（子类必须实现，家族解码逻辑差异所在）：
        将各检测层输出解码并融合为单输出等价数组
        Args:
            branch_outputs: 各检测层输出张量列表（引擎输出），每项shape为
                (batchsize,C,h,w)或(batchsize,C,h*w)
        Returns:
            每张图像的融合输出数组列表（与批量输出逐张切片形态一致，
            各家族列布局与单输出融合格式一致）
        """
        raise NotImplementedError

    @abstractmethod
    def encode(self,preds):
        """
        这是检测结果编码的抽象函数（子类必须实现，输出格式差异所在）
        Args:
            preds: 保留框的预测数组（[x1,y1,x2,y2,score,cls_id,附加列...]）
        Returns:
            检测结果字典列表；关键点等模型从中提取附加字段输出（如keypoints）
        """
        raise NotImplementedError

    def postprocess_decoded(self,preds,image_shape,restore_params=None):
        """
        这是统一预测数组的公共后处理尾部函数（解码后的统一流程）：
        尺度还原->NMS过滤->结果编码
        Args:
            preds: 统一预测数组（[x1,y1,x2,y2,score,cls_id,附加列...]）
            image_shape: 原始图像尺度(height,width)
            restore_params: 缩放逆变换参数(gain_h,gain_w,pad_h,pad_w)，
                预处理逐图给出（逐图w/h缩放系数与偏移系数），坐标还原按
                orig=(model-pad)/gain逐轴精确逆变换；None走旧口径
                （min gain单一还原，仅top_left锚定letterbox精确成立）
        Returns:
            检测结果字典列表
        """
        if len(preds) == 0:
            return []
        # 尺度还原（bbox列与声明的附加坐标列对统一还原，按restore_params逐轴）
        preds = rescale_coords(preds, self.model_hw, image_shape,
                               self.extra_coord_pairs,
                               restore_params=restore_params)
        # NMS过滤冗余框（过滤模式/抑制度量类型由实例配置nms_mode/iou_mode决定）
        indices = nms(preds, self.confidence_threshold, self.iou_threshold,
                      mode=self.nms_mode,
                      class_agnostic_threshold=self.class_agnostic_threshold,
                      iou_mode=self.iou_mode)
        if len(indices) == 0:
            return []
        # 对结果进行编码
        return self.encode(preds[indices])

    def postprocess_single(self,dets,image_shape,restore_params=None):
        """
        这是对单张图像的检测头输出做完整后处理的函数
        Args:
            dets: 模型输出结果张量
            image_shape: 原始图像尺度(height,width)
            restore_params: 缩放逆变换参数(gain_h,gain_w,pad_h,pad_w)
                （预处理逐图给出；语义见postprocess_decoded）
        Returns:
            检测结果字典列表
        """
        # 空输出保护：部分模型空帧时输出0个候选框，直接返回空结果
        # （否则argmax对空数组报ValueError）
        if np.shape(dets)[0] == 0:
            return []
        # 解码为统一预测数组（抽象方法，检测头差异所在）
        preds = self.decode(dets)
        return self.postprocess_decoded(preds, image_shape, restore_params)

    def postprocess(self,outputs,image_shapes,restore_params=None):
        """
        这是对批量图像的检测头输出做完整后处理的函数（总体后处理入口，
        兼容单输出与多输出两种模型形态）
        Args:
            outputs: 单输出模型为批量输出张量，shape为(batchsize,框数,分量数)；
                     多输出原始模型为各检测层输出张量列表，
                     每项shape为(batchsize,C,h,w)或(batchsize,C,h*w)
            image_shapes: 图像尺度数组，shape为(batchsize,2)
            restore_params: 逐图缩放逆变换参数数组，shape为(batchsize,4)
                （每行(gain_h,gain_w,pad_h,pad_w)——预处理逐图给出的w/h
                缩放系数与偏移系数；None走旧口径单一gain还原）
        Returns:
            检测结果列表，每张图像一个条目（无检测目标时为[]）
        """
        preds = []
        # 多输出原始模型：先经multioutputs_decode融合为单输出等价数组
        # （逐图像一项，子类实现家族解码逻辑），再走单输出后处理管线
        if isinstance(outputs, (list, tuple)):
            outputs = self.multioutputs_decode(outputs)
        for i in range(len(image_shapes)):
            # 单独处理一张图像的检测头输出（restore_params按图取行，
            # restore_params为None时逐图传None走旧口径）
            _restore_params = None if restore_params is None else restore_params[i]
            pred = self.postprocess_single(outputs[i], image_shapes[i],
                                           _restore_params)
            preds.append(pred)
        return preds
