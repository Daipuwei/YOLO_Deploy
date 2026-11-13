# -*- coding: utf-8 -*-
# @Time    : 2026/9/10 08:44
# @Author  : DaiPuWei
# @Email   : daipuwei@qq.com
# @File    : dflyolo.py
# @Software: PyCharm

"""
    这是定义DFL检测头YOLO模型（yolov6/v8/v9/v10/v11/v12/v13/v26）的脚本。

    模块内容：
        1. DFLYOLO：DFL检测头YOLO模型类，在类内完成三块装配：
           推理引擎（engine_type配置）+ DFL家族预处理
           （keep_aspect_ratio等比例缩放→BGR转RGB→无归一化（div255等已融入ONNX）→
           按布局转NCHW）+ DFLYOLO风格后处理（输出已为xyxy坐标+类别最大分打分+
           自适应num_classes+4/num_classes+5（scores_sum）输出格式+
           keep_aspect_ratio尺度还原+整体过滤NMS），检测流程由基类DetectionModel提供；
        2. yolov6/yolov8/yolov9/yolov10/yolov11/yolov12/yolov13/yolov26/ppyoloe(...)：
           模型工厂函数，注册进MODEL_REGISTRY（注册名即函数名），
           均完全复用DFLYOLO类，由model.build.build_model按名称装配。
"""

from ..build import MODEL_REGISTRY
from ..base_model import DetectionModel
from ...engine import build_engine
from ...utils import resolve_config_path
from ...preprocess import build_preprocessor
from ...postprocess import build_postprocessor
from ..detection_utils import get_classes

class DFLYOLO(DetectionModel):
    """
    这是DFL检测头YOLO模型类（类内完成三块装配，检测流程由基类DetectionModel提供）
    """

    def __init__(self,logger,cfg,gpu_id=0,**kwargs):
        """
        这是DFL检测头YOLO模型的初始化函数
        Args:
            logger: 日志类实例
            cfg: 参数配置字典（engine_type/engine_model_path/class_name_path/
                 confidence_threshold/iou_threshold/is_normalize/bbox_format等，
                 is_normalize缺省True；DFL导出div255已融入ONNX时需显式False，
                 bbox_format缺省xyxy，cxcywh导出需显式设置，
                 nms_mode缺省class_agnostic：整体过滤，可选per_class两阶段按类过滤）
            gpu_id: gpu设备号，默认为0
            kwargs: 自定义参数字典（其余键注入模型属性）
        """
        # 装配推理引擎（engine_type配置）
        engine = build_engine(logger, cfg, gpu_id=gpu_id)
        # 装配DFL家族预处理（input_shape与布局均来自引擎判定结果；
        # 无归一化：div255等已融入ONNX的模型无需再归一化）
        input_shapes = engine.get_input_shape()
        if len(input_shapes) != 1:
            raise ValueError("检测模型暂不支持多输入模型，引擎输入数量为{0}"
                             .format(len(input_shapes)))
        # is_normalize从配置读取（DetectionModel.is_normalize），缺省True；
        # 注意：DFL家族导出惯例div255已融入ONNX，使用此类导出时
        # 必须在配置中显式设置is_normalize=False
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
        # 装配DFLYOLO风格后处理（自适应nc+4/nc+5含scores_sum输出；
        # bbox_format从配置读取，缺省xyxy，cxcywh导出需显式设置）。
        # 类别名相对路径统一走resolve_config_path（绝对原样；相对CWD优先、
        # 仓库根兜底），非仓库根目录运行时仍可解析配置模板中的./model_data路径
        class_names = get_classes(resolve_config_path(cfg["class_name_path"]))
        bbox_format = cfg.get("bbox_format", "xyxy")
        output_layout = cfg.get("output_layout", "grid_first")
        nms_mode = cfg.get("nms_mode", "class_agnostic")
        # per_class第二阶段整体过滤阈值：缺省None用iou_threshold必须执行；
        # 显式配0=禁用、恢复"保留跨类重复框"的旧行为；>0=自定义阈值
        class_agnostic_threshold = cfg.get("class_agnostic_threshold", None)
        # 多输出原始模型解码参数（交后处理器multioutputs_decode使用，
        # 仅多输出模型生效，单输出不受影响；各检测层stride由model_hw与分支
        # 特征图hw自动推导、reg_max与scores_sum由特征图通道数与类别数自动
        # 判断、sigmoid由with_score_sum自动决定，均无需配置）
        postprocessor = build_postprocessor("dflyolo_postprocess",
                                            cfg["confidence_threshold"], cfg["iou_threshold"],
                                            class_names, (preprocessor.h, preprocessor.w),
                                            bbox_format=bbox_format,
                                            output_layout=output_layout,
                                            nms_mode=nms_mode,
                                            class_agnostic_threshold=class_agnostic_threshold)
        super(DFLYOLO, self).__init__(logger, engine, preprocessor=preprocessor,
                                      postprocessor=postprocessor,
                                      model_type=cfg["model_type"], **kwargs)
        self.logger.info("初始化{0}检测模型成功".format(self.model_type))


@MODEL_REGISTRY.register()
def yolov6(logger,cfg,**kwargs):
    """
    这是初始化yolov6检测模型的工厂函数（完全复用DFLYOLO类）
    Args:
        logger: 日志类实例
        cfg: 参数配置字典
        kwargs: 自定义参数字典（gpu_id等，透传给模型类）
    Returns:
        DFLYOLO检测模型实例（yolov6注册名）
    """
    return DFLYOLO(logger,cfg,**kwargs)


@MODEL_REGISTRY.register()
def yolov8(logger,cfg,**kwargs):
    """
    这是初始化yolov8检测模型的工厂函数（完全复用DFLYOLO类）
    Args:
        logger: 日志类实例
        cfg: 参数配置字典
        kwargs: 自定义参数字典（gpu_id等，透传给模型类）
    Returns:
        DFLYOLO检测模型实例（yolov8注册名）
    """
    return DFLYOLO(logger,cfg,**kwargs)


@MODEL_REGISTRY.register()
def yolov9(logger,cfg,**kwargs):
    """
    这是初始化yolov9检测模型的工厂函数（完全复用DFLYOLO类）
    Args:
        logger: 日志类实例
        cfg: 参数配置字典
        kwargs: 自定义参数字典（gpu_id等，透传给模型类）
    Returns:
        DFLYOLO检测模型实例（yolov9注册名）
    """
    return DFLYOLO(logger,cfg,**kwargs)


@MODEL_REGISTRY.register()
def yolov10(logger,cfg,**kwargs):
    """
    这是初始化yolov10检测模型的工厂函数（完全复用DFLYOLO类）
    Args:
        logger: 日志类实例
        cfg: 参数配置字典
        kwargs: 自定义参数字典（gpu_id等，透传给模型类）
    Returns:
        DFLYOLO检测模型实例（yolov10注册名）
    """
    return DFLYOLO(logger,cfg,**kwargs)


@MODEL_REGISTRY.register()
def yolov11(logger,cfg,**kwargs):
    """
    这是初始化yolov11检测模型的工厂函数（完全复用DFLYOLO类）
    Args:
        logger: 日志类实例
        cfg: 参数配置字典
        kwargs: 自定义参数字典（gpu_id等，透传给模型类）
    Returns:
        DFLYOLO检测模型实例（yolov11注册名）
    """
    return DFLYOLO(logger,cfg,**kwargs)


@MODEL_REGISTRY.register()
def yolov12(logger,cfg,**kwargs):
    """
    这是初始化yolov12检测模型的工厂函数（完全复用DFLYOLO类）
    Args:
        logger: 日志类实例
        cfg: 参数配置字典
        kwargs: 自定义参数字典（gpu_id等，透传给模型类）
    Returns:
        DFLYOLO检测模型实例（yolov12注册名）
    """
    return DFLYOLO(logger,cfg,**kwargs)


@MODEL_REGISTRY.register()
def yolov13(logger,cfg,**kwargs):
    """
    这是初始化yolov13检测模型的工厂函数（完全复用DFLYOLO类）
    Args:
        logger: 日志类实例
        cfg: 参数配置字典
        kwargs: 自定义参数字典（gpu_id等，透传给模型类）
    Returns:
        DFLYOLO检测模型实例（yolov13注册名）
    """
    return DFLYOLO(logger,cfg,**kwargs)


@MODEL_REGISTRY.register()
def yolo26(logger,cfg,**kwargs):
    """
    这是初始化yolov26检测模型的工厂函数（完全复用DFLYOLO类）
    Args:
        logger: 日志类实例
        cfg: 参数配置字典
        kwargs: 自定义参数字典（gpu_id等，透传给模型类）
    Returns:
        DFLYOLO检测模型实例（yolov26注册名）
    """
    return DFLYOLO(logger,cfg,**kwargs)


@MODEL_REGISTRY.register()
def ppyoloe(logger,cfg,**kwargs):
    """
    这是初始化ppyoloe检测模型的工厂函数（完全复用DFLYOLO类）
    Args:
        logger: 日志类实例
        cfg: 参数配置字典
        kwargs: 自定义参数字典（gpu_id等，透传给模型类）
    Returns:
        DFLYOLO检测模型实例（ppyoloe注册名）
    """
    return DFLYOLO(logger,cfg,**kwargs)
