# -*- coding: utf-8 -*-
# @Time    : 2026/9/11 08:09
# @Author  : DaiPuWei
# @Email   : daipuwei@qq.com
# @File    : yolo_metric_utils.py
# @Software: PyCharm

"""
    这是定义Ultralytics风格YOLO口径检测性能评估工具函数的脚本
    （纯numpy实现，无需ultralytics依赖）。

    模块内容：
        1. match_predictions(pred_cls, true_cls, iou, iou_thresholds)：逐图像
           贪心匹配（预测按置信度降序，同类别且IoU达标者优先匹配）；
        2. compute_ap(recall, precision)：PR曲线梯形积分；
        3. ap_per_class(tp, conf, pred_cls, target_cls)：逐类别P/R/F1/AP统计；
        4. compute_yolo_metrics(detections, gts, class_names, ...)：总体入口，
           返回整体指标字典（P/R/mAP50/mAP50-95/F1/混淆矩阵）、
           逐类别表格与混淆矩阵表格（行=预测类别，列=真实类别）；
        5. format_confusion_matrix(confusion_matrix, class_names)：混淆矩阵
           表格格式化函数（行=预测类别，列=真实类别，最后一行为漏检、
           最后一列为误检）；
        6. ConfusionMatrix：混淆矩阵统计类（含背景行/列）。
        与COCOeval口径的差异：单IoU阈值贪心匹配 + PR曲线梯形积分
        （COCO为101点插值），两者数值存在微小差异，属正常口径差异。
        P/R/F1按最大F1工作点取值（IoU=0.5的PR曲线1000点插值，
        取平滑后F1曲线峰值点，与ultralytics口径一致，不受conf阈值影响）；
        mAP仍按全部预测积分，受conf阈值影响（与ultralytics一致）。
"""

import numpy as np
from tabulate import tabulate

def _bbox_iou_matrix(boxes1,boxes2):
    """
    这是计算两组xyxy框的IoU矩阵的函数
    Args:
        boxes1: 框数组，shape为(框数1,4)
        boxes2: 框数组，shape为(框数2,4)
    Returns:
        IoU矩阵，shape为(框数1,框数2)
    """
    x11, y11, x12, y12 = np.split(boxes1, 4, axis=1)
    x21, y21, x22, y22 = np.split(boxes2, 4, axis=1)
    xa = np.maximum(x11, x21.T)
    ya = np.maximum(y11, y21.T)
    xb = np.minimum(x12, x22.T)
    yb = np.minimum(y12, y22.T)
    inter = np.maximum(0, xb - xa) * np.maximum(0, yb - ya)
    area1 = (x12 - x11) * (y12 - y11)
    area2 = (x22 - x21) * (y22 - y21)
    union = area1 + area2.T - inter
    return inter / (union + 1e-16)


def match_predictions(pred_cls,true_cls,iou,iou_thresholds):
    """
    这是将预测框与真实框按类别与IoU进行贪心匹配的函数
    （复刻ultralytics的match_predictions语义，纯numpy实现）：
    预测按置信度降序依次匹配，每个预测选择与其类别相同、未被匹配且
    IoU最大的真实框；在最小IoU阈值下匹配成功即占用该真实框。
    Args:
        pred_cls: 预测类别数组（按置信度降序排列），shape为(框数,)
        true_cls: 真实类别数组，shape为(真实框数,)
        iou: 预测与真实的IoU矩阵，shape为(框数,真实框数)
        iou_thresholds: IoU阈值数组，shape为(阈值数,)。
            必须单调递增：第0列为最小阈值（约定IoU=0.5，供ap_per_class
            的最大F1工作点与PR曲线取列），贪心占位逻辑（correct[:,0]）
            依赖该顺序——传入降序/乱序阈值会使同一GT被重复计TP
            （recall可>1）且工作点取错列，故入口显式校验非法即报错
    Returns:
        correct: 各预测在各IoU阈值下是否匹配成功的布尔数组，
                 shape为(框数,阈值数)
    """
    # 阈值顺序校验（修复：曾隐含升序假设不校验，外部传降序阈值时
    # GT占位与工作点取列双双错位，指标静默失真）
    iou_thresholds = np.asarray(iou_thresholds, dtype=np.float64)
    if iou_thresholds.ndim != 1 or len(iou_thresholds) == 0:
        raise ValueError("iou_thresholds必须为一维非空数组，实际{0}"
                         .format(iou_thresholds))
    if len(iou_thresholds) > 1 and np.any(np.diff(iou_thresholds) <= 0):
        raise ValueError("iou_thresholds必须单调递增（第0列为最小阈值/IoU=0.5），"
                         "实际{0}".format(list(iou_thresholds)))
    num_preds = len(pred_cls)
    num_thrs = len(iou_thresholds)
    correct = np.zeros((num_preds, num_thrs), dtype=bool)
    matched = np.zeros(len(true_cls), dtype=bool)
    for i in range(num_preds):
        # 同类别且未被匹配的真实框
        candidates = np.where((true_cls == pred_cls[i]) & (~matched))[0]
        if len(candidates) == 0:
            continue
        # 贪心：取IoU最大的候选框
        best_j = candidates[np.argmax(iou[i, candidates])]
        # 各IoU阈值下达到阈值即视为匹配成功
        for t, thr in enumerate(iou_thresholds):
            if iou[i, best_j] >= thr:
                correct[i, t] = True
        # 在最小阈值下匹配成功则占用该真实框
        if correct[i, 0]:
            matched[best_j] = True
    return correct


def _smooth(y,f=0.05):
    """
    这是对曲线做箱式滤波平滑的函数（复刻ultralytics的smooth实现，
    用于最大F1工作点选取前消除曲线抖动）
    Args:
        y: 输入曲线数组
        f: 滤波窗口比例，默认为0.05
    Returns:
        平滑后的曲线（长度与输入一致）
    """
    nf = round(len(y) * f * 2) // 2 + 1  # 窗口长度（奇数）
    pad = np.ones(nf // 2)
    yp = np.concatenate((pad * y[0], y, pad * y[-1]), 0)
    return np.convolve(yp, np.ones(nf) / nf, mode="valid")


def compute_ap(recall,precision):
    """
    这是按PR曲线梯形积分计算AP的函数（复刻ultralytics）
    Args:
        recall: 召回率数组（按置信度降序）
        precision: 精确率数组（按置信度降序）
    Returns:
        AP值
    """
    # 在曲线首尾补点并做后向最大单调化
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([1.0], precision, [0.0]))
    mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))
    # 梯形积分
    ap = np.trapz(mpre, mrec)
    return ap


def ap_per_class(tp,conf,pred_cls,target_cls):
    """
    这是按类别统计precision/recall/AP/F1的函数
    （PR曲线梯形积分口径，复刻ultralytics；类别集合为np.unique(target_cls)
    ——只含有GT的类别，ultralytics口径，仅有预测而无GT的类不进入整体均值
    分母（其误检计入该类Precision行与混淆矩阵误检列，避免重复计罚）；无GT
    类别时按全零口径返回空数组，由调用方将整体指标记0）
    Args:
        tp: 匹配成功布尔数组，shape为(框数,阈值数)。
            列序必须对应match_predictions的升序iou_thresholds
            （第0列为最小阈值/IoU=0.5工作点列，P/R/F1取值依赖此约定）
        conf: 预测置信度数组
        pred_cls: 预测类别数组
        target_cls: 真实类别数组
    Returns:
        p/r/f1: 逐类别精确率/召回率/F1数组
            （按最大F1工作点取值，与ultralytics口径一致：
            IoU=0.5的PR曲线1000点插值，取平滑后F1曲线峰值点；
            该口径下P/R/F1不受conf阈值影响，反映模型最佳工作点性能）
        ap: 逐类别AP数组，shape为(类别数,阈值数)
        unique_classes: 类别数组（0起索引的真实类别；ultralytics口径
            只含有GT的类别，仅有预测而无GT的类不在此列，无GT时为空数组）
    """
    # 按置信度降序排序（PR曲线的前提）
    i = np.argsort(-conf)
    tp, conf, pred_cls = tp[i], conf[i], pred_cls[i]
    # 唯一类别：只取有GT的类别（ultralytics口径np.unique(target_cls)）。
    # 仅有预测而无GT的类不进入整体均值的分母——其误检已计入该类Precision行
    # 与混淆矩阵误检列，若再进均值分母会重复计罚，使整体指标系统性偏低
    unique_classes = np.unique(target_cls[target_cls >= 0]).astype(int)
    if len(unique_classes) == 0:
        # 无GT类别：PR曲线无定义，按全零口径返回（整体指标由调用方记0）
        return (np.zeros(0), np.zeros(0), np.zeros(0),
                np.zeros((0, tp.shape[1])), unique_classes)
    max_cls = unique_classes.max()
    nt = np.bincount(target_cls[target_cls >= 0].astype(int), minlength=max_cls + 1)
    nc = unique_classes.shape[0]
    num_thrs = tp.shape[1]
    ap = np.zeros((nc, num_thrs))
    p = np.zeros(nc)
    r = np.zeros(nc)
    f1 = np.zeros(nc)
    # P/R/F1按最大F1工作点取值：IoU=0.5的PR曲线1000点插值
    x = np.linspace(0, 1, 1000)
    p_curve = np.zeros((nc, 1000))
    r_curve = np.zeros((nc, 1000))
    for ci, c in enumerate(unique_classes):
        mask = pred_cls == c
        n_p = mask.sum()
        n_l = nt[c]
        if n_p == 0 or n_l == 0:
            continue
        tpc = tp[mask].cumsum(0)
        fpc = (1 - tp[mask]).cumsum(0)
        recall = tpc / (n_l + 1e-16)
        precision = tpc / (tpc + fpc + 1e-16)
        for j in range(num_thrs):
            ap[ci, j] = compute_ap(recall[:, j], precision[:, j])
        # IoU=0.5曲线按置信度插值到1000点（左端补齐recall=0/precision=1）
        r_curve[ci] = np.interp(-x, -conf[mask], recall[:, 0], left=0)
        p_curve[ci] = np.interp(-x, -conf[mask], precision[:, 0], left=1)
    # 最大F1工作点：类别平均F1曲线经箱式滤波平滑后取峰值点
    f1_curve = 2 * p_curve * r_curve / (p_curve + r_curve + 1e-16)
    best_idx = _smooth(f1_curve.mean(0), 0.1).argmax()
    p, r, f1 = p_curve[:, best_idx], r_curve[:, best_idx], f1_curve[:, best_idx]
    return p, r, f1, ap, unique_classes


class ConfusionMatrix(object):
    """
    这是混淆矩阵统计的类（复刻ultralytics的ConfusionMatrix语义：
    矩阵行对应预测类别、列对应真实类别，最后一行为/列对应背景；
    匹配按IoU进行、不限制类别，用于观察类别间的错检模式）
    """

    def __init__(self,nc,conf_threshold=0.25,iou_threshold=0.45):
        """
        这是混淆矩阵统计类的初始化函数
        Args:
            nc: 类别数量
            conf_threshold: 置信度阈值，低于该值的预测不计入，默认为0.25
            iou_threshold: IoU匹配阈值，默认为0.45
        """
        self.matrix = np.zeros((nc + 1, nc + 1))
        self.nc = nc
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold

    def process_batch(self,pred_boxes,pred_conf,pred_cls,gt_boxes,gt_cls):
        """
        这是按单张图像更新混淆矩阵的函数
        Args:
            pred_boxes: 预测框数组（xyxy），shape为(框数,4)
            pred_conf: 预测置信度数组
            pred_cls: 预测类别数组
            gt_boxes: 真实框数组（xyxy），shape为(真实框数,4)
            gt_cls: 真实类别数组
        Returns:
            更新后的混淆矩阵
        """
        pred_boxes = np.array(pred_boxes)
        pred_conf = np.array(pred_conf)
        pred_cls = np.array(pred_cls, dtype=int)
        gt_boxes = np.array(gt_boxes)
        gt_cls = np.array(gt_cls, dtype=int)
        # 闭区间边界：encode时分数4位舍入，恰好等于阈值的框不应漏计
        keep = pred_conf >= self.conf_threshold
        pred_boxes, pred_cls = pred_boxes[keep], pred_cls[keep]
        if len(pred_boxes) == 0:
            # 无预测：全部真实框计入背景行（漏检），不得静默丢弃
            for c in gt_cls:
                self.matrix[self.nc, c] += 1
            return self.matrix
        if len(gt_boxes) == 0:
            # 无真实框：全部预测计入背景列
            for c in pred_cls:
                self.matrix[c, self.nc] += 1
            return self.matrix
        iou = _bbox_iou_matrix(gt_boxes, pred_boxes)  # (真实框数,预测框数)
        # 候选匹配（仅按IoU，不限制类别）：先按IoU降序，再保证每个预测
        # 最多对应一个真实框、每个真实框最多对应一个预测
        x = np.where(iou > self.iou_threshold)
        matches = np.stack((x[0], x[1], iou[x[0], x[1]]), axis=1)
        matches = matches[matches[:, 2].argsort()[::-1]]
        matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
        matches = matches[matches[:, 2].argsort()[::-1]]
        matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
        matched_pred = set()
        matched_gt = set()
        for m in matches:
            i, j = int(m[0]), int(m[1])
            self.matrix[pred_cls[j], gt_cls[i]] += 1
            matched_pred.add(j)
            matched_gt.add(i)
        # 未匹配的预测计入背景列，未匹配的真实框计入背景行
        for j in range(len(pred_cls)):
            if j not in matched_pred:
                self.matrix[pred_cls[j], self.nc] += 1
        for i in range(len(gt_cls)):
            if i not in matched_gt:
                self.matrix[self.nc, gt_cls[i]] += 1
        return self.matrix


def format_confusion_matrix(confusion_matrix,class_names):
    """
    这是生成混淆矩阵表格字符串的函数
    表格语义：行=预测类别，列=真实类别；
    最后一行为漏检（真实框未被任何预测匹配，计入背景行），
    最后一列为误检（预测框未匹配到任何真实框，计入背景列）。
    Args:
        confusion_matrix: 混淆矩阵数组，shape为(类别数+1,类别数+1)
        class_names: 类别名称数组
    Returns:
        混淆矩阵表格字符串（整数计数，行=预测类别，列=真实类别）
    """
    matrix = np.asarray(confusion_matrix, dtype=int)
    if matrix.shape != (len(class_names) + 1, len(class_names) + 1):
        raise ValueError("混淆矩阵shape({0})与类别数({1})不符，期望({2},{2})"
                         .format(matrix.shape, len(class_names), len(class_names) + 1))
    rows = []
    for row_idx, name in enumerate(class_names):
        rows.append([name] + matrix[row_idx].tolist())
    # 最后一行：漏检（真实框未被匹配）
    rows.append(["背景(漏检)"] + matrix[-1].tolist())
    table = tabulate(rows,
                     headers=["预测\\真实"] + class_names + ["背景(误检)"],
                     tablefmt="pipe", numalign="center", intfmt=",")
    return table


def compute_yolo_metrics(detections,gts,class_names,iou_thresholds=None,
                          conf_threshold=None):
    """
    这是计算Ultralytics风格YOLO口径评估指标的函数（纯numpy实现）。
    整体指标（all行与metrics字典）只按有GT的类别求均值（ultralytics口径，
    仅有预测而无GT的类不进均值分母，此时打印[WARN]提示，其误检体现在
    逐类Precision行与混淆矩阵误检列）；无GT类别时整体指标记0
    Args:
        detections: 检测结果列表，每项含bbox(x1y1x2y2)/category_id/score/image_id
        gts: 真实标注列表，每项含bbox(x1y1x2y2)/category_id/image_id
        class_names: 类别名称数组
        iou_thresholds: IoU阈值数组，默认None使用0.5:0.95十档
        conf_threshold: 混淆矩阵统计的置信度阈值，默认None使用0.25；
            建议传评测所用的confidence_threshold，保证混淆矩阵与
            P/R/mAP等指标的统计口径一致（混淆矩阵匹配IoU沿用0.45惯例）
    Returns:
        metrics: 整体指标字典（precision/recall/mAP50/mAP50_95/f1/
                 confusion_matrix）
        metrics_table: 逐类别指标表格字符串
        confusion_table: 混淆矩阵表格字符串（行=预测类别，列=真实类别）
    """
    if iou_thresholds is None:
        iou_thresholds = np.round(np.arange(0.5, 1.0, 0.05), 2)
    # 按图像分组
    detections_by_image = {}
    for det in detections:
        detections_by_image.setdefault(det["image_id"], []).append(det)
    gts_by_image = {}
    for gt in gts:
        gts_by_image.setdefault(gt["image_id"], []).append(gt)
    # 逐图像贪心匹配并累积统计量
    tp_list = []
    conf_list = []
    pred_cls_list = []
    # 混淆矩阵按评测conf阈值统计（缺省None时用ConfusionMatrix默认0.25），
    # 保证与P/R/mAP等指标口径一致（匹配IoU沿用0.45惯例）
    confusion = (ConfusionMatrix(len(class_names))
                 if conf_threshold is None
                 else ConfusionMatrix(len(class_names), conf_threshold=conf_threshold))
    # 遍历有预测图像与有GT图像的并集：仅含GT无预测的图像也要计入
    # 混淆矩阵的漏检行（FN统计完整性）
    for image_id in set(detections_by_image.keys()) | set(gts_by_image.keys()):
        dets = detections_by_image.get(image_id, [])
        gts_img = gts_by_image.get(image_id, [])
        pred_boxes = np.array([d["bbox"] for d in dets])
        gt_boxes = np.array([g["bbox"] for g in gts_img])
        pred_cls = np.array([d["category_id"] for d in dets])
        gt_cls = np.array([g["category_id"] for g in gts_img])
        conf = np.array([d["score"] for d in dets])
        # 按置信度降序排序（贪心匹配前提）
        order = np.argsort(-conf)
        pred_boxes, pred_cls, conf = pred_boxes[order], pred_cls[order], conf[order]
        # 空预测保护：该图有GT但无预测（漏检）时匹配数组取空；
        # 空GT保护：该图有预测但无真实框（误检）时IoU矩阵取空
        # （_bbox_iou_matrix对(0,)形状的np.split会AxisError）
        if len(pred_boxes) == 0:
            correct = np.zeros((0, len(iou_thresholds)), dtype=bool)
        else:
            if len(gt_boxes) == 0:
                gt_boxes = np.zeros((0, 4))
                iou = np.zeros((len(pred_boxes), 0))
            else:
                iou = _bbox_iou_matrix(pred_boxes, gt_boxes)
            correct = match_predictions(pred_cls, gt_cls, iou, iou_thresholds)
        tp_list.append(correct)
        conf_list.append(conf)
        pred_cls_list.append(pred_cls)
        # 混淆矩阵统计
        confusion.process_batch(pred_boxes, conf, pred_cls, gt_boxes, gt_cls)
    if len(tp_list) == 0:
        # 无任何预测：指标全零
        metrics = {"precision": 0.0, "recall": 0.0, "f1": 0.0, "mAP50": 0.0, "mAP50_95": 0.0,
                   "confusion_matrix": confusion.matrix}
        table = tabulate([], headers=["类别", "标注数", "预测数", "Precision", "Recall","F1",
                                      "mAP@0.5", "mAP@0.5:0.95"], tablefmt="pipe")
        confusion_table = format_confusion_matrix(confusion.matrix, class_names)
        return metrics, table, confusion_table
    tp = np.concatenate(tp_list, 0)
    conf = np.concatenate(conf_list, 0)
    pred_cls = np.concatenate(pred_cls_list, 0)
    target_cls = np.array([g["category_id"] for g in gts])
    p, r, f1, ap, unique_classes = ap_per_class(tp, conf, pred_cls, target_cls)
    # 整体指标：按有GT类别均值（ultralytics口径；无GT类别时数组为空，记0）
    precision = float(p.mean()) if len(p) > 0 else 0.0
    recall = float(r.mean()) if len(r) > 0 else 0.0
    map50 = float(ap[:, 0].mean()) if len(ap) > 0 else 0.0
    map50_95 = float(ap.mean()) if len(ap) > 0 else 0.0
    f1_mean = float(f1.mean()) if len(f1) > 0 else 0.0
    metrics = {
        "precision": precision,
        "recall": recall,
        "mAP50": map50,
        "mAP50_95": map50_95,
        "f1": f1_mean,
    }
    metrics["confusion_matrix"] = confusion.matrix
    # 预防告警：仅有预测而无GT的类别不计入整体均值（ultralytics口径），
    # 其误检已体现在逐类Precision行与混淆矩阵误检列，此处提示避免误读
    pred_only_classes = np.setdiff1d(np.unique(pred_cls), unique_classes)
    if len(pred_only_classes) > 0:
        pred_only_names = [class_names[c] if 0 <= c < len(class_names) else str(c)
                           for c in pred_only_classes]
        print("[WARN] 以下类别在评测集中无GT，未计入YOLO口径整体指标均值"
              "（误检已计入混淆矩阵最后一列）：{0}".format(pred_only_names))
    # 逐类别表格
    nt = np.bincount(target_cls[target_cls >= 0].astype(int),
                      minlength=len(class_names))
    rows = [
        ["all",int(np.sum(nt)),
         int(np.sum([int((pred_cls == c).sum()) for c in unique_classes])),
         precision,recall,f1_mean,map50,map50_95]
    ]
    for ci, c in enumerate(unique_classes):
        rows.append([class_names[c], int(nt[c]), int((pred_cls == c).sum()),
                     p[ci], r[ci], f1[ci],ap[ci, 0], ap[ci].mean()])
    metrics_table = tabulate(rows, headers=["类别", "标注数", "预测数", "Precision", "Recall","F1",
                                    "mAP@0.5", "mAP@0.5:0.95"],
                     tablefmt="pipe", numalign="center",
                     floatfmt=".4f")
    # 混淆矩阵表格（行=预测类别，列=真实类别，最后一行为漏检、最后一列为误检）
    confusion_table = format_confusion_matrix(confusion.matrix, class_names)

    return metrics, metrics_table, confusion_table
