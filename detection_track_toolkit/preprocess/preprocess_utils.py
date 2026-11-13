# -*- coding: utf-8 -*-
# @Time    : 2026/9/10 08:26
# @Author  : DaiPuWei
# @Email   : daipuwei@qq.com
# @File    : preprocess_utils.py
# @Software: PyCharm

"""
    这是定义预处理独立工具函数的脚本。

    模块内容：
        1. letterbox_params(image_shape, resize_shape, pad_mode)：计算letterbox
           缩放与填充参数的纯函数（缩放后尺寸与整像素填充量，两种填充锚定
           模式top_left/center共用；填充与坐标还原共用同一组参数，
           保证互为精确逆变换）；
        2. resize_keep_aspect_ratio(image, resize_shape=(640, 640),
           color=(114, 114, 114), pad_mode='top_left')：等比例缩放并填充像素
           的函数（历史名称letterbox，由utils.detection_utils移入并改名；
           pad_mode选择填充锚定：'top_left'缩放图贴左上角、填充全在右下
           （缺省，坐标还原无偏移）；'center'居中填充（ultralytics口径，
           坐标还原需减去填充偏移，见BasePreprocessor.compute_restore_params）；
           图像尺寸与目标尺寸一致时直接返回原图零拷贝）；
        3. resize(image, resize_shape=(640, 640))：直接拉伸填满
           目标尺寸的函数（不保持宽高比；图像尺寸与目标尺寸一致时
           直接返回原图零拷贝；坐标还原需按h/w双轴独立gain，
           见BasePreprocessor.compute_restore_params）。
        两者resize_shape均为(height,width)约定（与仓库model_hw一致）。
        本模块函数均为无状态纯函数，可独立调用；BasePreprocessor内部也调用
        本模块函数。
"""

import cv2


def letterbox_params(image_shape,resize_shape=(640, 640),pad_mode="top_left"):
    """
    这是计算letterbox缩放与填充参数的纯函数（填充执行与坐标还原共用，
    保证两者互为精确逆变换；整像素填充量，pad不对称时左右/上下差≤1px）
    Args:
        image_shape: 原始图像尺寸(height,width)
        resize_shape: 目标尺寸(height,width)，默认为(640,640)
        pad_mode: 填充锚定模式，默认为'top_left'（缩放图贴左上角，
            填充全在右下，坐标还原无偏移）；'center'居中填充
            （ultralytics口径，-0.1/+0.1取整修正，坐标还原需减填充偏移）
    Returns:
        (new_h, new_w, top, bottom, left, right)：缩放后尺寸与四侧整像素填充量
    """
    if pad_mode not in ("top_left", "center"):
        raise ValueError("未知的填充锚定模式'{0}'，支持top_left或center"
                         .format(pad_mode))
    h, w = image_shape[0:2]
    input_h,input_w = resize_shape
    # 等比例缩放比例与缩放后尺寸（整像素，下限钳到1）。
    # 修复：极端宽高比下（如1×1280、2×2560条带）h*r<0.5时round得0，
    # new_h=0使cv2.resize(dsize含0)崩溃、且gain_h=new_h/h=0导致后处理
    # 还原除零；钳到1后图像被压到单像素条带+填充，gain保持非零可安全还原
    r = min(input_w / w,input_h / h)
    new_h, new_w = max(1, int(round(h * r))), max(1, int(round(w * r)))
    dh,dw = input_h-new_h,input_w-new_w
    if pad_mode == "center":
        # 居中填充：各分一半，-0.1取整修正避免x.5时银行家舍入抖动
        top, left = int(round(dh / 2 - 0.1)), int(round(dw / 2 - 0.1))
    else:
        # 左上锚定：填充全在右下
        top, left = 0, 0
    bottom, right = dh-top, dw-left
    return new_h, new_w, top, bottom, left, right


def resize_keep_aspect_ratio(image,resize_shape=(640, 640),color=(114, 114, 114),
                              pad_mode="top_left"):
    """
    这是等比例缩放并填充像素的函数（原letterbox；填充锚定由pad_mode选择：
    'top_left'缩放图贴左上角、填充全在右下（缺省）；'center'居中填充
    （ultralytics口径）。填充参数由letterbox_params统一计算，
    坐标还原侧经BasePreprocessor.compute_restore_params取同一组参数）
    Args:
        image: 图像，opencv格式
        resize_shape: 目标尺寸(height,width)，默认为(640,640)
        color: 填充像素值，默认为(114,114,114)
        pad_mode: 填充锚定模式，默认为'top_left'（无偏移还原）或'center'
                  （居中填充，还原需减填充偏移，见letterbox_params）
    Returns:
        等比例缩放并填充到目标尺寸后的图像
        （图像尺寸与目标尺寸一致时直接返回原图，零拷贝跳过缩放填充）
    """
    h, w, _ = image.shape
    input_h,input_w = resize_shape
    # 尺寸一致直接返回原图：跳过缩放与填充（视频检测时输入分辨率
    # 恰为模型输入尺寸的场景收益明显）
    if h == input_h and w == input_w:
        return image
    # 计算等比例缩放与填充参数（两种锚定模式共用）并执行缩放填充
    new_h, new_w, top, bottom, left, right = letterbox_params((h, w), resize_shape,
                                                              pad_mode)
    resize_image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    resize_image = cv2.copyMakeBorder(resize_image, top, bottom, left, right,
                                      cv2.BORDER_CONSTANT, value=color)  # add border
    return resize_image


def resize(image,resize_shape=(640, 640)):
    """
    这是直接拉伸填满目标尺寸的函数（cv2.resize的封装，不保持宽高比）
    Args:
        image: 图像，opencv格式
        resize_shape: 目标尺寸(height,width)，默认为(640,640)
    Returns:
        直接拉伸到目标尺寸后的图像
        （图像尺寸与目标尺寸一致时直接返回原图，零拷贝跳过缩放）
    """
    h, w = image.shape[0:2]
    input_h,input_w = resize_shape
    # 尺寸一致直接返回原图：跳过缩放
    if h == input_h and w == input_w:
        return image
    return cv2.resize(image, (input_w, input_h), interpolation=cv2.INTER_LINEAR)
