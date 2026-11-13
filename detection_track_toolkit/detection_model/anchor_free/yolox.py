# -*- coding: utf-8 -*-
# @Time    : 2026/9/10 08:44
# @Author  : DaiPuWei
# @Email   : daipuwei@qq.com
# @File    : yolox.py
# @Software: PyCharm

"""
    这是定义YOLOX模型的脚本。

    模块内容：
        1. YOLOX：YOLOX检测模型类，在类内完成三块装配：
           推理引擎（engine_type配置）+ YOLOX风格预处理
           （keep_aspect_ratio等比例缩放→BGR直通→无归一化→按布局转NCHW）+
           YOLOX风格后处理（输出已为xyxy坐标+obj过滤×obj×cls打分+
           keep_aspect_ratio尺度还原+整体过滤NMS），检测流程由基类DetectionModel提供；
        2. yolox(...)：模型工厂函数，注册进MODEL_REGISTRY（注册名'yolox'），
           由model.build.build_model按名称装配。
"""

from ..build import MODEL_REGISTRY
from ..base_model import DetectionModel
from ...engine import build_engine
from ...utils import resolve_config_path
from ...preprocess import build_preprocessor
from ...postprocess import build_postprocessor
from ..detection_utils import get_classes

class YOLOX(DetectionModel):
    """
    这是YOLOX检测模型类（类内完成三块装配，检测流程由基类DetectionModel提供）
    """

    def __init__(self,logger,cfg,gpu_id=0,**kwargs):
        """
        这是YOLOX检测模型的初始化函数
        Args:
            logger: 日志类实例
            cfg: 参数配置字典（engine_type/engine_model_path/class_name_path/
                 confidence_threshold/iou_threshold/bbox_format/nms_mode等，
                 bbox_format缺省xyxy，cxcywh导出需显式设置，
                 nms_mode缺省class_agnostic：整体过滤，可选per_class两阶段按类过滤）
            gpu_id: gpu设备号，默认为0
            kwargs: 自定义参数字典（其余键注入模型属性）
        """
        # 装配推理引擎（engine_type配置）
        engine = build_engine(logger, cfg, gpu_id=gpu_id)
        # 装配YOLOX风格预处理（input_shape与布局均来自引擎判定结果）
        input_shapes = engine.get_input_shape()
        if len(input_shapes) != 1:
            raise ValueError("检测模型暂不支持多输入模型，引擎输入数量为{0}"
                             .format(len(input_shapes)))
        # 缩放方式与letterbox填充锚定透传（缺省keep_aspect_ratio+top_left；
        # resize拉伸/居中填充的坐标还原由restore_params逐图精确逆变换支持；
        # 修复：此前装配不透传resize_mode，配置写了也被静默忽略）
        resize_mode = cfg.get("resize_mode", "keep_aspect_ratio")
        pad_mode = cfg.get("pad_mode", "top_left")
        preprocessor = build_preprocessor("yolox_preprocess", input_shapes[0],
                                          is_nchw=engine.get_is_nchw(),
                                          resize_mode=resize_mode,
                                          pad_mode=pad_mode)
        # 装配YOLOX风格后处理（bbox_format从配置读取，缺省xyxy，
        # cxcywh导出需显式设置）。
        # 类别名相对路径统一走resolve_config_path（绝对原样；相对CWD优先、
        # 仓库根兜底），非仓库根目录运行时仍可解析配置模板中的./model_data路径
        class_names = get_classes(resolve_config_path(cfg["class_name_path"]))
        bbox_format = cfg.get("bbox_format", "xyxy")
        output_layout = cfg.get("output_layout", "grid_first")
        with_sigmoid = cfg.get("with_sigmoid", True)
        nms_mode = cfg.get("nms_mode", "class_agnostic")
        # per_class第二阶段整体过滤阈值：缺省None用iou_threshold必须执行；
        # 显式配0=禁用、恢复"保留跨类重复框"的旧行为；>0=自定义阈值
        class_agnostic_threshold = cfg.get("class_agnostic_threshold", None)
        postprocessor = build_postprocessor("yolox_postprocess",
                                            cfg["confidence_threshold"], cfg["iou_threshold"],
                                            class_names, (preprocessor.h, preprocessor.w),
                                            bbox_format=bbox_format,
                                            output_layout=output_layout,
                                            with_sigmoid=with_sigmoid,
                                            nms_mode=nms_mode,
                                            class_agnostic_threshold=class_agnostic_threshold)
        super(YOLOX, self).__init__(logger, engine, preprocessor=preprocessor,
                                    postprocessor=postprocessor,
                                    model_type=cfg["model_type"], **kwargs)
        # 多输出原始模型解码已内聚为YOLOXPostprocessor.multioutputs_decode
        # （postprocess中自动调用，无需模型侧装配）
        self.logger.info("初始化YOLOX检测模型成功")


@MODEL_REGISTRY.register()
def yolox(logger,cfg,**kwargs):
    """
    这是初始化YOLOX检测模型的工厂函数
    Args:
        logger: 日志类实例
        cfg: 参数配置字典
        kwargs: 自定义参数字典（gpu_id等，透传给模型类）
    Returns:
        YOLOX检测模型实例
    """
    model = YOLOX(logger,cfg,**kwargs)
    return model
