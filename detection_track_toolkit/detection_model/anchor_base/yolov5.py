# -*- coding: utf-8 -*-
# @Time    : 2026/9/10 08:44
# @Author  : DaiPuWei
# @Email   : daipuwei@qq.com
# @File    : yolov5.py
# @Software: PyCharm

"""
    这是定义YOLOv5与YOLOv7模型的脚本。

    模块内容：
        1. YOLOv5：YOLOv5检测模型类，在类内完成三块装配：
           推理引擎（engine_type配置）+ YOLOv5风格预处理
           （keep_aspect_ratio等比例缩放→BGR转RGB→除以255归一化→按布局转NCHW）+
           YOLOv5风格后处理（xywh→xyxy坐标解码+obj×cls打分+keep_aspect_ratio尺度还原+
           整体过滤NMS），检测流程由基类DetectionModel提供；
        2. yolov5(...)：YOLOv5模型工厂函数，注册进MODEL_REGISTRY
           （注册名'yolov5'），由model.build.build_model按名称装配；
        3. yolov7(...)：YOLOv7模型工厂函数，注册进MODEL_REGISTRY
           （注册名'yolov7'），完全复用YOLOv5类
           （YOLOv7与YOLOv5的预处理/后处理装配完全一致）。
"""

import numpy as np

from ..build import MODEL_REGISTRY
from ..base_model import DetectionModel
from ...engine import build_engine
from ...utils import resolve_config_path
from ...preprocess import build_preprocessor
from ...postprocess import build_postprocessor
from ..detection_utils import get_anchors
from ..detection_utils import get_classes

class YOLOv5(DetectionModel):
    """
    这是YOLOv5检测模型类（类内完成三块装配，检测流程由基类DetectionModel提供）
    """

    def __init__(self,logger,cfg,gpu_id=0,**kwargs):
        """
        这是YOLOv5检测模型的初始化函数
        Args:
            logger: 日志类实例
            cfg: 参数配置字典（engine_type/engine_model_path/class_name_path/
                 confidence_threshold/iou_threshold/is_normalize/nms_mode等，
                 is_normalize缺省True：输入除以255归一化，
                 nms_mode缺省class_agnostic：整体过滤，可选per_class两阶段按类过滤）
            gpu_id: gpu设备号，默认为0
            kwargs: 自定义参数字典（其余键注入模型属性）
        """
        # 装配推理引擎（engine_type配置）
        engine = build_engine(logger, cfg, gpu_id=gpu_id)
        # 装配YOLOv5风格预处理（input_shape与布局均来自引擎判定结果）
        input_shapes = engine.get_input_shape()
        if len(input_shapes) != 1:
            raise ValueError("检测模型暂不支持多输入模型，引擎输入数量为{0}"
                             .format(len(input_shapes)))
        # is_normalize从配置读取（DetectionModel.is_normalize），缺省True：
        # YOLOv5风格导出惯例输入需除以255归一化
        is_normalize = cfg.get("is_normalize", True)
        # 缩放方式与letterbox填充锚定透传（缺省keep_aspect_ratio+top_left；
        # resize拉伸/居中填充的坐标还原由restore_params逐图精确逆变换支持）
        resize_mode = cfg.get("resize_mode", "keep_aspect_ratio")
        pad_mode = cfg.get("pad_mode", "top_left")
        preprocessor = build_preprocessor("yolov5_preprocess", input_shapes[0],
                                          is_nchw=engine.get_is_nchw(),
                                          is_normalize=is_normalize,
                                          resize_mode=resize_mode,
                                          pad_mode=pad_mode)
        # 装配YOLOv5风格后处理（nms_mode从配置读取，缺省class_agnostic整体过滤）
        # 配置内相对路径统一走resolve_config_path（绝对原样；相对CWD优先、
        # 仓库根兜底），非仓库根目录运行时仍可解析配置模板中的./model_data路径
        class_names = get_classes(resolve_config_path(cfg["class_name_path"]))
        nms_mode = cfg.get("nms_mode", "class_agnostic")
        # per_class第二阶段整体过滤阈值：缺省None用iou_threshold必须执行；
        # 显式配0=禁用、恢复"保留跨类重复框"的旧行为；>0=自定义阈值
        class_agnostic_threshold = cfg.get("class_agnostic_threshold", None)
        # 多输出原始模型解码参数（anchors经txt文件读取，方式同类别名文件；
        # 交后处理器multioutputs_decode使用，仅多输出模型生效，单输出不受影响；
        # 各检测层stride由model_hw与分支特征图hw自动推导，无需配置）
        # anchors缺省取仓库根model_data/coco_anchors.txt并经resolve_config_path
        # 解析（修复：曾写死机器特定绝对路径/data/Milesight/...，换机器部署
        # 时直接FileNotFoundError）
        with_sigmoid = cfg.get("with_sigmoid", True)
        anchors_path = cfg.get("anchors_path", "./model_data/coco_anchors.txt")
        anchors = get_anchors(resolve_config_path(anchors_path))
        # 输出层数由engine输出节点个数决定：get_anchors返回(anchor数,2)，
        # 在此按层数拆分为(层数,每层anchor数,2)
        num_outputs = len(engine.output_shapes)
        if len(anchors) % num_outputs != 0:
            raise ValueError("anchors数量({0})与引擎输出层数({1})不匹配，无法均分"
                             .format(len(anchors), num_outputs))
        anchors = np.asarray(anchors, dtype=np.float32) \
            .reshape(num_outputs, -1, 2).tolist()
        postprocessor = build_postprocessor("yolov5_postprocess",
                                            cfg["confidence_threshold"], cfg["iou_threshold"],
                                            class_names, (preprocessor.h, preprocessor.w),
                                            with_sigmoid=with_sigmoid,
                                            anchors=anchors,
                                            nms_mode=nms_mode,
                                            class_agnostic_threshold=class_agnostic_threshold)
        super(YOLOv5, self).__init__(logger, engine, preprocessor=preprocessor,
                                     postprocessor=postprocessor,
                                     model_type=cfg["model_type"], **kwargs)
        self.logger.info("初始化{0}检测模型成功".format(self.model_type))


@MODEL_REGISTRY.register()
def yolov5(logger,cfg,**kwargs):
    """
    这是初始化YOLOv5检测模型的工厂函数
    Args:
        logger: 日志类实例
        cfg: 参数配置字典
        kwargs: 自定义参数字典（gpu_id等，透传给模型类）
    Returns:
        YOLOv5检测模型实例
    """
    model = YOLOv5(logger,cfg,**kwargs)
    return model


@MODEL_REGISTRY.register()
def yolov7(logger,cfg,**kwargs):
    """
    这是初始化YOLOv7检测模型的工厂函数（YOLOv7与YOLOv5的
    预处理/后处理装配完全一致，完全复用YOLOv5类）
    Args:
        logger: 日志类实例
        cfg: 参数配置字典
        kwargs: 自定义参数字典（gpu_id等，透传给模型类）
    Returns:
        YOLOv5检测模型实例（yolov7注册名）
    """
    model = YOLOv5(logger,cfg,**kwargs)
    return model
