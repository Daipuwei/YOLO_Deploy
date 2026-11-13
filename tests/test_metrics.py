# -*- coding: utf-8 -*-
"""
    YOLO口径评估指标（最大F1工作点语义）单元测试
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from detection_track_toolkit.utils.metric_utils.yolo_metric_utils import (
    ap_per_class, compute_yolo_metrics)


def test_ap_per_class_max_f1_operating_point():
    """P/R/F1取最大F1工作点而非曲线末点（与ultralytics口径一致）：
    2个GT被2个高置信TP命中，另有10个超低分FP紧跟其后——
    末点口径P=2/12=0.167，最大F1工作点口径P应远高于末点（≈0.98）。
    注：F1=1.0仅在x=0.8单点成立（FP紧跟TP之后无平台），
    平滑取峰后工作点P略低于1.0属与ultralytics一致的正常行为"""
    num_thrs = 10
    # 预测按置信度降序：2个TP（高置信），10个FP（超低置信）
    tp = np.zeros((12, num_thrs), dtype=bool)
    tp[0:2, :] = True
    conf = np.array([0.9, 0.8] + [0.005] * 10, dtype=np.float64)
    pred_cls = np.zeros(12, dtype=int)
    target_cls = np.zeros(2, dtype=int)
    p, r, f1, ap, classes = ap_per_class(tp, conf, pred_cls, target_cls)
    assert list(classes) == [0]
    # 语义判别：工作点P远高于末点P（0.167），接近1.0
    assert p[0] > 0.95, "P应取最大F1工作点(>0.95)，实际{0}".format(p[0])
    assert r[0] > 0.95, "R应取最大F1工作点(>0.95)，实际{0}".format(r[0])
    # mAP不受影响：全部预测积分，超低分FP排在曲线尾部，AP≈1.0
    assert ap[0, 0] > 0.99 and ap[0].mean() > 0.99, ap[0]


def test_match_predictions_threshold_order_validation():
    """iou_thresholds必须单调递增（回归：曾不校验，传降序阈值时GT占位
    correct[:,0]与工作点取列双双错位——同一GT可重复计TP、recall可>1）"""
    from detection_track_toolkit.utils.metric_utils.yolo_metric_utils import (
        match_predictions)
    pred_cls = np.array([0])
    true_cls = np.array([0])
    iou = np.array([[0.6]])
    # 降序阈值：报错
    try:
        match_predictions(pred_cls, true_cls, iou, np.array([0.75, 0.5]))
        assert False, "降序阈值应报错"
    except ValueError as e:
        assert "单调递增" in str(e), e
    # 乱序阈值：报错
    try:
        match_predictions(pred_cls, true_cls, iou, np.array([0.5, 0.95, 0.75]))
        assert False, "乱序阈值应报错"
    except ValueError:
        pass
    # 空数组与重复阈值：报错
    for bad in (np.array([]), np.array([0.5, 0.5])):
        try:
            match_predictions(pred_cls, true_cls, iou, bad)
            assert False, "非法阈值应报错"
        except ValueError:
            pass
    # 合法升序：正常匹配
    correct = match_predictions(pred_cls, true_cls, iou,
                                np.array([0.5, 0.75]))
    assert correct.shape == (1, 2) and correct[0, 0] and not correct[0, 1]


def test_compute_yolo_metrics_end_to_end():
    """单类合成数据：2 GT，预测2 TP(高置信)+2 FP(低置信)，
    最大F1工作点口径下P=R=F1=1.0"""
    dets = [
        {"image_id": 0, "category_id": 0, "score": 0.9, "bbox": [10, 10, 40, 40]},
        {"image_id": 0, "category_id": 0, "score": 0.8, "bbox": [50, 50, 90, 90]},
        {"image_id": 0, "category_id": 0, "score": 0.005, "bbox": [1, 1, 5, 5]},
        {"image_id": 0, "category_id": 0, "score": 0.004, "bbox": [95, 95, 99, 99]},
    ]
    gts = [
        {"image_id": 0, "category_id": 0, "bbox": [10, 10, 40, 40]},
        {"image_id": 0, "category_id": 0, "bbox": [50, 50, 90, 90]},
    ]
    metrics, table, confusion_table = compute_yolo_metrics(dets, gts, ["person"])
    # 工作点口径：P/R/F1接近1.0（略低于1.0属平滑取峰的正常行为）
    assert metrics["precision"] > 0.95, metrics
    assert metrics["recall"] > 0.95, metrics
    assert metrics["f1"] > 0.95, metrics
    assert metrics["mAP50"] > 0.99, metrics
    # 混淆矩阵表格
    assert "背景(误检)" in confusion_table and "背景(漏检)" in confusion_table


def test_compute_yolo_metrics_excludes_no_gt_classes():
    """整体指标只按有GT类别求均值（ultralytics口径np.unique(target_cls)）：
    person完美检测+1个无GT的dog误检 → P/R/F1/mAP应为1.0而非0.5
    （回归：曾因类别集合取pred∪gt并集，全零行进入均值被拉低一半）"""
    dets = [
        {"image_id": 0, "category_id": 0, "score": 0.95, "bbox": [10, 10, 40, 40]},
        {"image_id": 0, "category_id": 1, "score": 0.90, "bbox": [60, 60, 80, 80]},
    ]
    gts = [
        {"image_id": 0, "category_id": 0, "bbox": [10, 10, 40, 40]},
    ]
    metrics, table, confusion_table = compute_yolo_metrics(dets, gts, ["person", "dog"])
    assert metrics["precision"] > 0.95, metrics
    assert metrics["recall"] > 0.95, metrics
    assert metrics["f1"] > 0.95, metrics
    assert metrics["mAP50"] > 0.99, metrics
    assert metrics["mAP50_95"] > 0.99, metrics
    # 逐类表格按ultralytics口径只列有GT的类（dog行不出现）；
    # dog误检体现在预防告警与混淆矩阵误检列
    assert "dog" not in table
    assert "背景(误检)" in confusion_table


def test_ap_per_class_no_gt_returns_zero_metrics():
    """无GT类别边界：按全零口径返回空数组，调用方均值记0且不产生NaN"""
    num_thrs = 10
    tp = np.zeros((1, num_thrs), dtype=bool)
    p, r, f1, ap, classes = ap_per_class(tp, np.array([0.9]),
                                         np.zeros(1, dtype=int),
                                         np.zeros(0, dtype=int))
    assert len(classes) == 0, classes
    assert len(p) == 0 and len(r) == 0 and len(f1) == 0
    assert ap.shape == (0, num_thrs), ap.shape
    metrics, _, _ = compute_yolo_metrics(
        [{"image_id": 0, "category_id": 0, "score": 0.9, "bbox": [1, 1, 5, 5]}],
        [], ["person"])
    assert metrics["precision"] == 0.0 and metrics["mAP50"] == 0.0, metrics


def test_eval_per_class_all_row_masks_na_sentinel():
    """COCO逐类表格all行剔除N/A哨兵（-1）：cat完美+cat2无GT →
    all行为1.0（回归：曾为mean([1.0,-1.0])=-0.0000），cat2行保持-1显示，
    all行与pycocotools全局mAP口径一致"""
    import tempfile
    from detection_track_toolkit.utils.metric_utils.coco_metric_utils import (
        save_coco_json, eval_coco, eval_per_class)
    tmp_dir = tempfile.mkdtemp()
    gt_path = os.path.join(tmp_dir, "gt.json")
    pred_path = os.path.join(tmp_dir, "pred.json")
    image_infos = [{"file_name": "a.jpg", "id": 0, "width": 100, "height": 100}]
    gt = [{"image_id": 0, "iscrowd": 0, "bbox": [10, 10, 30, 30], "area": 900,
           "category_id": 0, "id": 1}]
    pred = [{"image_id": 0, "iscrowd": 0, "bbox": [10, 10, 30, 30], "area": 900,
             "category_id": 0, "id": 1, "score": 0.9}]
    class_names = ["cat", "cat2"]
    save_coco_json(gt_path, image_infos, gt, class_names)
    save_coco_json(pred_path, image_infos, pred, class_names)
    evaluator, global_map, global_map50, _ = eval_coco(gt_path, pred_path)
    class_stats, table = eval_per_class(evaluator, class_names)
    # all行=有效类（cat）的真实值，不再被-1哨兵平均污染
    assert abs(class_stats[0] - 1.0) < 1e-9, class_stats
    assert abs(class_stats[1] - 1.0) < 1e-9, class_stats
    # 与pycocotools全局统计口径一致（回归前自相矛盾）
    assert abs(class_stats[0] - global_map) < 1e-9
    # cat2行保持-1（N/A显示惯例）
    assert "-1.0000" in table


def _run_all():
    tests = [(k, v) for k, v in list(globals().items())
             if k.startswith("test_") and callable(v)]
    fails = 0
    for name, fn in sorted(tests):
        try:
            fn()
            print("PASS", name)
        except Exception as e:
            fails += 1
            print("FAIL", name, "->", type(e).__name__, str(e)[:200])
    print("=" * 40, "fails:", fails)
    return fails


if __name__ == "__main__":
    sys.exit(1 if _run_all() else 0)
