# -*- coding: utf-8 -*-
# @Time    : 2026/9/10 05:27
# @Author  : DaiPuWei
# @Email   : daipuwei@qq.com
# @File    : postprocess_utils.py
# @Software: PyCharm

"""
    这是定义后处理独立工具函数的脚本。

    模块内容：
        1. iou_xyxy(box, boxes, mode='iou')：计算一个框与多个框的重叠度量
           （xyxy格式，几何正确，零面积框按0处理），mode支持iou交并比（缺省）、
           giou广义交并比（包围框面积惩罚）、diou距离交并比（中心点距离惩罚）；
        2. nms(preds, score_threshold, iou_threshold, mode='class_agnostic',
           class_agnostic_threshold=None, iou_mode='iou')：统一NMS函数
           （纯numpy自实现），mode支持class_agnostic整体过滤（缺省）、
           per_class按类别过滤后整体过滤（两阶段，第二阶段必须执行）；
           iou_mode选择抑制度量类型（iou/giou/diou）；
        3. rescale_coords(preds, model_hw, image_shape, extra_coord_pairs=())：
           统一坐标还原函数，bbox列(0:4)恒还原，extra_coord_pairs声明的附加坐标
           列对做相同gain还原与越界裁剪；
        4. scale_coords(coords, input_shape, image_shape)：检测框坐标还原到原始
           图像尺度并做越界截断的函数（由utils.detection_utils移入并合并原
           clip_coords，原处副本为未迁移模型保留，迁移完成后删除）；
        5. sigmoid(x)：sigmoid激活函数（向量化广播执行，多输出解码中
           ONNX未带sigmoid时使用）；
        6. make_grid(h, w, offset=0)：网格坐标生成函数（(2,grids)布局：首行x、
           次行y，与检测层输出(C,grids)布局对齐，解码时免转置；offset为整体
           偏移量，缺省0，如DFL解码传0.5得网格中心点）；
        7. softmax(x, axis)：按轴做softmax的函数（DFL分布回归用，稳定数值实现）；
        8. dfl_decode(bboxes, reg_max)：DFL分布回归解码函数（reg_max>1时
           softmax加权（softmax分布回归@arange）得4个偏移量；
           reg_max=1时无分布区间，仅reshape直接输出）；
        9. dist2bbox(offsets, grid_center, bbox_format='xyxy')：偏移量转bbox
           坐标函数（网格中心点与偏移量差/和得网格单元尺度坐标，
           bbox_format支持xyxy/cxcywh两种格式输出；stride还原由调用方完成）。
        本模块函数均为无状态纯函数，可独立调用；BasePostprocessor内部也调用
        本模块函数，保证直接调用与类内行为一致。
"""

import numpy as np


def validate_iou_mode(mode):
    """这是校验iou度量类型合法性的函数"""
    if mode not in ("iou", "giou", "diou"):
        raise ValueError("未知的iou模式'{0}'，支持iou/giou/diou"
                         .format(mode))


def iou_xyxy(box,boxes,mode="iou"):
    """
    这是计算一个框与多个框的重叠度量的函数（xyxy格式，几何正确）
    Args:
        box: 单个框坐标[x1,y1,x2,y2]
        boxes: 多个框坐标数组，shape为(框数,4)
        mode: 重叠度量类型，默认为'iou'：'iou'交并比；'giou'广义交并比
            （引入包围框面积惩罚）；'diou'距离交并比（引入中心点距离惩罚）
    Returns:
        重叠度量数组（零面积框按0处理）
    """
    validate_iou_mode(mode)
    xx1 = np.maximum(box[0], boxes[:, 0])
    yy1 = np.maximum(box[1], boxes[:, 1])
    xx2 = np.minimum(box[2], boxes[:, 2])
    yy2 = np.minimum(box[3], boxes[:, 3])
    inter = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
    area = (box[2] - box[0]) * (box[3] - box[1])
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    union = area + areas - inter
    valid = union > 0
    # 公共量：包围框（两个框的最小外接矩形）
    cx1 = np.minimum(box[0], boxes[:, 0])
    cy1 = np.minimum(box[1], boxes[:, 1])
    cx2 = np.maximum(box[2], boxes[:, 2])
    cy2 = np.maximum(box[3], boxes[:, 3])
    area_c = (cx2 - cx1) * (cy2 - cy1)
    diag_c2 = (cx2 - cx1) ** 2 + (cy2 - cy1) ** 2
    # 公共量：中心点距离平方
    center_x1 = (box[0] + box[2]) / 2.0
    center_y1 = (box[1] + box[3]) / 2.0
    center_x2 = (boxes[:, 0] + boxes[:, 2]) / 2.0
    center_y2 = (boxes[:, 1] + boxes[:, 3]) / 2.0
    dist_c2 = (center_x2 - center_x1) ** 2 + (center_y2 - center_y1) ** 2
    # 公共量：iou基础值（零面积框按0处理），各度量在分支中叠加惩罚项
    iou = np.zeros_like(inter)
    iou[valid] = inter[valid] / union[valid]
    if mode == "iou":
        return iou
    elif mode == "giou":
        # giou = iou - (C - 并集) / C，C为最小外接矩形面积
        giou = np.zeros_like(inter)
        giou[valid] = iou[valid] - (area_c[valid] - union[valid]) / area_c[valid]
        return giou
    else:
        # diou = iou - 中心点距离平方 / 包围框对角线平方
        diou = np.zeros_like(inter)
        diou[valid] = iou[valid] - dist_c2[valid] / diag_c2[valid]
        return diou


def validate_nms_mode(mode):
    """这是校验NMS过滤模式合法性的函数"""
    if mode not in ("class_agnostic", "per_class"):
        raise ValueError("未知的nms模式'{0}'，支持class_agnostic/per_class"
                         .format(mode))


def nms_greedy(preds,score_threshold,iou_threshold,iou_mode):
    """
    这是贪心NMS的核心函数（score降序依次保留最高分框并抑制重叠框）
    Args:
        preds: 预测数组（[x1,y1,x2,y2,score,cls_id,附加列...]）
        score_threshold: 分数阈值（分数不高于该阈值的框被过滤）
        iou_threshold: 抑制度量阈值（超过该阈值则抑制）
        iou_mode: 重叠度量类型（iou/giou/diou，见iou_xyxy）
    Returns:
        保留框的下标数组
    """
    order = np.argsort(-preds[:, 4])
    keep = []
    while len(order) > 0:
        first = order[0]
        # 分数降序排列：当前最高分不高于阈值则其余框均不高于阈值，直接结束
        if preds[first, 4] <= score_threshold:
            break
        keep.append(first)
        if len(order) == 1:
            break
        # 抑制与当前框的重叠度量超过阈值的其余框
        rest = order[1:]
        metric = iou_xyxy(preds[first, 0:4], preds[rest, 0:4], iou_mode)
        order = rest[metric < iou_threshold]
    return keep


def nms(preds,score_threshold,iou_threshold,mode="class_agnostic",
        class_agnostic_threshold=None,iou_mode="iou"):
    """
    这是NMS过滤冗余框的统一函数（纯numpy自实现）
    Args:
        preds: 预测数组（[x1,y1,x2,y2,score,cls_id,附加列...]）
        score_threshold: 分数阈值（分数不高于该阈值的框被过滤）
        iou_threshold: 抑制度量阈值（超过该阈值则抑制）
        mode: NMS过滤模式，默认为'class_agnostic'（整体过滤，与原始cv2语义一致）；
              'per_class'为按类别过滤后整体过滤（两阶段：先按类别去重，不同类别
              互不抑制，再对所有保留框做一次整体过滤，第二阶段阈值缺省与
              iou_threshold相同，可经class_agnostic_threshold单独指定，如0.9）
        class_agnostic_threshold: per_class模式第二阶段整体过滤阈值，
            缺省None即用iou_threshold做全局过滤（第二阶段必须执行，
            跨类别重复框按iou_threshold抑制）；>0按该阈值全局过滤；
            0或负显式禁用全局过滤（仅按类别过滤，跨类别重复框保留）；
            class_agnostic模式忽略
        iou_mode: 重叠度量类型，默认为'iou'，候选有giou/diou
            （度量定义与阈值语义见iou_xyxy）
    Returns:
        保留框的下标数组
    """
    validate_nms_mode(mode)
    validate_iou_mode(iou_mode)
    # 整体过滤模式：所有类别混在一起做一次贪心NMS
    if mode == "class_agnostic":
        return np.array(nms_greedy(preds, score_threshold, iou_threshold, iou_mode),
                        dtype=np.int64)
    # 第一阶段：按类别分组NMS，不同类别的框不互相抑制（修复类别无关NMS造成的漏检）
    cls_id = preds[:, 5].astype(np.int64)
    keep = []
    for class_id in np.unique(cls_id):
        class_pos = np.where(cls_id == class_id)[0]
        _keep = nms_greedy(preds[class_pos], score_threshold, iou_threshold, iou_mode)
        keep.extend(class_pos[int(idx)] for idx in _keep)
    keep = np.array(keep, dtype=np.int64)
    # 第二阶段：整体过滤，对第一阶段保留框跨类别去重，
    # 去除同一目标被多个类别同时预测的近似重复框。
    # 阈值取值：None（缺省）用iou_threshold必须执行；>0用指定阈值；
    # <=0显式禁用（仅按类别过滤，跨类别重复框全部保留）——显式配0即
    # 恢复"保留跨类重复框"的旧行为
    global_threshold = iou_threshold if class_agnostic_threshold is None \
        else class_agnostic_threshold
    if global_threshold <= 0:
        return keep
    _keep = nms_greedy(preds[keep], score_threshold, global_threshold, iou_mode)
    keep = keep[_keep]
    return keep


def scale_coords(bboxes,input_shape,image_shape):
    """
    这是将检测框坐标还原到原始图像尺度的函数
    Args:
        bboxes: 检测框数组，格式为(x1,y1,x2,y2)
        input_shape: 模型输入尺度(height,width)
        image_shape: 原始图像尺度(height,width)
    Returns:
        还原到原始图像尺度后的检测框数组
    """
    # 将检测框坐标还原到原始图像尺度
    gain = min(input_shape[0] / image_shape[0], input_shape[1] / image_shape[1])
    bboxes /= gain
    # 避免坐标越界
    bboxes[:, 0] = np.clip(bboxes[:, 0], 0, image_shape[1])   # x1
    bboxes[:, 1] = np.clip(bboxes[:, 1], 0, image_shape[0])   # y1
    bboxes[:, 2] = np.clip(bboxes[:, 2], 0, image_shape[1])   # x2
    bboxes[:, 3] = np.clip(bboxes[:, 3], 0, image_shape[0])   # y2
    return bboxes


def rescale_coords(preds,model_hw,image_shape,extra_coord_pairs=(),
                   restore_params=None):
    """
    这是预测数组坐标尺度还原的统一函数（bbox列与附加坐标列对做相同还原）。
    还原口径优先取预处理逐图给出的restore_params（逐图w/h的缩放系数与
    偏移系数，与预处理实际变换互为精确逆变换）：
    统一逆变换 orig=(model-pad)/gain按轴独立——letterbox两种填充锚定
    （top_left pad=0；center需减居中填充偏移）与resize拉伸（双轴独立gain）
    均正确还原。restore_params为None时走旧口径（按min(model_hw/图像hw)
    单一gain反推，仅对top_left锚定letterbox成立，兼容外部直接调用）。
    Args:
        preds: 预测数组（[x1,y1,x2,y2,score,cls_id,附加列...]）
        model_hw: 模型输入尺寸(height,width)（restore_params缺省时的旧口径用）
        image_shape: 原始图像尺度(height,width)
        extra_coord_pairs: 附加坐标列对（如关键点x,y列号），如((6,7),(8,9))
        restore_params: 缩放逆变换参数(gain_h,gain_w,pad_h,pad_w)，
            由BasePreprocessor.compute_restore_params逐图计算并经
            DetectionModel.detect透传；None走旧口径

    Returns:
        尺度还原后的预测数组

    """
    if restore_params is not None:
        gain_h, gain_w, pad_h, pad_w = restore_params
    else:
        # 旧口径（兼容外部直接调用）：单一gain= min(model_hw/图像hw)、无偏移，
        # 仅对top_left锚定letterbox精确成立
        gain_h = gain_w = min(model_hw[0] / image_shape[0],
                              model_hw[1] / image_shape[1])
        pad_h = pad_w = 0.0
    # bbox列按轴还原（x轴列0/2、y轴列1/3）并越界裁剪
    preds[:, 0] = np.clip((preds[:, 0] - pad_w) / gain_w, 0, image_shape[1])   # x1
    preds[:, 1] = np.clip((preds[:, 1] - pad_h) / gain_h, 0, image_shape[0])   # y1
    preds[:, 2] = np.clip((preds[:, 2] - pad_w) / gain_w, 0, image_shape[1])   # x2
    preds[:, 3] = np.clip((preds[:, 3] - pad_h) / gain_h, 0, image_shape[0])   # y2
    # 声明的附加坐标列对（关键点x,y等）做相同的按轴还原与越界裁剪
    for cx, cy in extra_coord_pairs:
        preds[:, cx] = np.clip((preds[:, cx] - pad_w) / gain_w, 0, image_shape[1])
        preds[:, cy] = np.clip((preds[:, cy] - pad_h) / gain_h, 0, image_shape[0])
    return preds


def sigmoid(x):
    """
    这是sigmoid激活函数（向量化广播执行，对数组整体一次激活，
    避免逐元素循环；多输出解码中ONNX未带sigmoid时使用）
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


def softmax(x,axis):
    """
    这是按轴做softmax的函数（DFL分布回归用，稳定数值实现）
    Args:
        x: 输入数组
        axis: softmax作用的轴
    Returns:
        softmax归一化后的数组（与输入同shape）
    """
    e = np.exp(x - x.max(axis=axis, keepdims=True))
    return e / e.sum(axis=axis, keepdims=True)


def dfl_decode(bboxes,reg_max):
    """
    这是DFL分布回归解码的函数（多输出原始模型bbox解码封装）：
    reg_max>1时做softmax加权（softmax分布回归@arange得4个偏移量）；
    reg_max=1时无分布区间可加权（softmax加权结果恒为0偏移，无意义），
    仅reshape直接输出原始回归值
    Args:
        bboxes: DFL回归logit，shape为(batchsize,4,reg_max,grids)
            （4个角顺序为[left,top,right,bottom]）
        reg_max: DFL回归区间数
    Returns:
        4个偏移量，shape为(batchsize,4,grids)
    """
    if reg_max == 1:
        # reg_max=1时无分布区间：仅reshape为(batchsize,4,grids)，
        # 此时通道即[l,t,r,b]四个偏移量
        return bboxes.reshape(bboxes.shape[0], 4, bboxes.shape[-1])
    index = np.arange(reg_max, dtype=np.float32).reshape(reg_max, 1)  # (reg_max,1)
    # 转置为(bs,grids,4,reg_max)后按最后一维softmax加权，dot得期望偏移量
    offsets = softmax(np.transpose(bboxes, (0, 3, 1, 2)), axis=3).dot(index)  # (bs,grids,4,1)
    offsets = np.transpose(offsets[:, :, :, 0], (0, 2, 1))  # 还原为(bs,4,grids)
    return offsets

def dist2bbox(offsets,grid_center,bbox_format="xyxy"):
    """
    这是偏移量转bbox坐标的函数（dist2bbox，多输出原始模型bbox解码封装）：
    网格中心点与偏移量差/和得网格单元尺度的bbox坐标（bbox_format参数
    支持xyxy与cxcywh两种格式输出；stride还原由调用方完成）
    Args:
        offsets: DFL回归偏移量，shape为(batchsize,4,grids)
            （4个角顺序为[left,top,right,bottom]）
        grid_center: 网格中心点坐标，shape为(2,grids)（首行x、次行y）
        bbox_format: bbox坐标布局，'xyxy'（缺省）或'cxcywh'
    Returns:
        bbox坐标（网格单元尺度），shape为(batchsize,4,grids)，
        布局[x1,y1,x2,y2]（xyxy）或[cx,cy,w,h]（cxcywh）
    """
    x1y1 = grid_center - offsets[:, :2]
    x2y2 = grid_center + offsets[:, 2:]
    if bbox_format == "cxcywh":
        # xyxy转cxcywh：中心点坐标与宽高
        bbox = np.concatenate([(x1y1 + x2y2) / 2.0, x2y2 - x1y1], axis=1)
    else:
        bbox = np.concatenate([x1y1, x2y2], axis=1)
    return bbox
