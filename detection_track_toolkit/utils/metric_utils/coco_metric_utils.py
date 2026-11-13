# -*- coding: utf-8 -*-
# @Time    : 2026/9/11 08:09
# @Author  : DaiPuWei
# @Email   : daipuwei@qq.com
# @File    : coco_metric_utils.py
# @Software: PyCharm

"""
    这是定义COCOeval口径检测性能评估工具函数的脚本。

    模块内容：
        1. summarize_eval(coco_eval, cat_id=None)：按类别（或整体）汇总COCOeval
           评估结果的函数（复刻pycocotools的summarize逻辑，支持单类别统计）；
        2. eval_coco(gt_json_path, pred_json_path)：执行COCOeval标准评估的函数，
           返回(评估器, mAP@0.5:0.95, mAP@0.5, mAR@0.5:0.95)；
        3. eval_per_class(coco_eval, class_names)：计算逐类别AP/AR指标并生成
           tabulate表格的函数，返回(类别均值数组, 表格字符串)；
        4. save_coco_json(json_path, image_infos, annotations, class_names)：
           保存COCO格式标注JSON的函数（gt与pred共用，含numpy类型序列化）。
        注意：COCO标注id必须非零（cocoapi将dtMatches中的0视为未匹配）。
"""

import json
import numpy as np

from tabulate import tabulate
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from ..json_utils import NpEncoder


def summarize_eval(coco_eval,cat_id=None):
    """
    这是汇总COCOeval评估结果的函数（复刻pycocotools的summarize逻辑，
    支持按类别统计；仅适用于默认参数设置）
    Args:
        coco_eval: COCOeval评估器实例（需已执行accumulate）
        cat_id: 类别id，默认为None统计所有类别
    Returns:
        stats: 10项汇总指标数组
        print_info: 汇总信息字符串
    """

    if not coco_eval.eval:
        raise Exception('Please run accumulate() first')

    def _summarize(ap=1, iouThr=None, areaRng='all', maxDets=100):
        p = coco_eval.params
        iStr = ' {:<18} {} @[ IoU={:<9} | area={:>6s} | maxDets={:>3d} ] = {:0.3f}'
        titleStr = 'Average Precision' if ap == 1 else 'Average Recall'
        typeStr = '(AP)' if ap == 1 else '(AR)'
        iouStr = '{:0.2f}:{:0.2f}'.format(p.iouThrs[0], p.iouThrs[-1]) \
            if iouThr is None else '{:0.2f}'.format(iouThr)

        aind = [i for i, aRng in enumerate(p.areaRngLbl) if aRng == areaRng]
        mind = [i for i, mDet in enumerate(p.maxDets) if mDet == maxDets]

        if ap == 1:
            # dimension of precision: [TxRxKxAxM]
            s = coco_eval.eval['precision']
            # IoU
            if iouThr is not None:
                t = np.where(iouThr == p.iouThrs)[0]
                s = s[t]

            if isinstance(cat_id, int):
                s = s[:, :, cat_id, aind, mind]
            else:
                s = s[:, :, :, aind, mind]

        else:
            # dimension of recall: [TxKxAxM]
            s = coco_eval.eval['recall']
            if iouThr is not None:
                t = np.where(iouThr == p.iouThrs)[0]
                s = s[t]

            if isinstance(cat_id, int):
                s = s[:, cat_id, aind, mind]
            else:
                s = s[:, :, aind, mind]

        if len(s[s > -1]) == 0:
            mean_s = -1
        else:
            mean_s = np.mean(s[s > -1])

        print_string = iStr.format(titleStr, typeStr, iouStr, areaRng, maxDets, mean_s)
        return mean_s, print_string

    stats, print_list = [0] * 10, [""] * 10
    stats[0], print_list[0] = _summarize(1)
    stats[1], print_list[1] = _summarize(1, iouThr=.5, maxDets=coco_eval.params.maxDets[2])
    stats[2], print_list[2] = _summarize(1, iouThr=.75, maxDets=coco_eval.params.maxDets[2])
    stats[3], print_list[3] = _summarize(1, iouThr=.95, maxDets=coco_eval.params.maxDets[2])
    stats[4], print_list[4] = _summarize(0, iouThr=.5, maxDets=coco_eval.params.maxDets[2])
    stats[5], print_list[5] = _summarize(0, iouThr=.75, maxDets=coco_eval.params.maxDets[2])
    stats[6], print_list[6] = _summarize(0, iouThr=.95, maxDets=coco_eval.params.maxDets[2])
    stats[7], print_list[7] = _summarize(0, maxDets=coco_eval.params.maxDets[0])
    stats[8], print_list[8] = _summarize(0, maxDets=coco_eval.params.maxDets[1])
    stats[9], print_list[9] = _summarize(0, maxDets=coco_eval.params.maxDets[2])

    print_info = "\n".join(print_list)

    return stats, print_info


def eval_coco(gt_json_path,pred_json_path):
    """
    这是执行COCOeval标准评估的函数
    Args:
        gt_json_path: 真实标签JSON路径（COCO格式）
        pred_json_path: 检测结果JSON路径（COCO格式）
    Returns:
        evaluator: COCOeval评估器实例（已执行evaluate/accumulate/summarize）
        map: mAP@0.5:0.95
        map50: mAP@0.5
        mar: mAR@0.5:0.95
    """
    # 加载标注与检测结果
    coco_gt = COCO(gt_json_path)
    coco_pred = COCO(pred_json_path)
    # 创建检测框评估器
    evaluator = COCOeval(coco_gt, coco_pred, iouType='bbox')
    evaluator.evaluate()
    evaluator.accumulate()
    evaluator.summarize()
    map, map50 = evaluator.stats[:2]
    mar = evaluator.stats[9]   # AR@0.5:0.95@maxDets=100（stats[8]为maxDets=10口径）
    return evaluator, map, map50, mar


def eval_per_class(coco_eval,class_names):
    """
    这是计算逐类别AP/AR指标并生成表格的函数。
    all行均值按列剔除N/A哨兵（-1，无GT类别的无定义条目）后仅对有效值求均值
    （与pycocotools全局统计s[s>-1]同口径，避免哨兵被当真实指标平均导致
    all行失真，如完美模型被无GT类拉成0甚至负值）；某列全无效时保持-1；
    逐类行保持COCO惯例原样显示（无GT类为-1）
    Args:
        coco_eval: COCOeval评估器实例（需已执行accumulate）
        class_names: 类别名称数组
    Returns:
        class_stats: all行指标数组（8项，与表格列顺序一致：
                     mAP@0.5:0.95、mAP@0.5、mAP@0.75、mAP@0.95、
                     mAR@0.5:0.95、mAR@0.5、mAR@0.75、mAR@0.95；
                     逐列剔除N/A哨兵-1后求均值，某列全无效时保持-1）
        coco_metric_table: tabulate表格字符串（含all行）
    """
    # 计算每个类别的AP和AR指标
    class_stats = []
    coco_metric_table = []
    for i in range(len(class_names)):
        stats, _ = summarize_eval(coco_eval, cat_id=i)
        ap50_95,ap50,ap75,ap95,ar50,ar75,ar95 = stats[0:7]
        ar50_95 = stats[9]   # AR@0.5:0.95@maxDets=100
        class_stats.append([ap50_95,ap50,ap75,ap95,ar50_95,ar50,ar75,ar95])
        coco_metric_table.append([class_names[i],ap50_95,ap50,ap75,ap95,ar50_95,ar50,ar75,ar95])
    class_stats = np.array(class_stats)
    # all行均值剔除N/A哨兵（-1，无GT类别的无定义条目）：AP/AR合法值域为[0,1]，
    # 与pycocotools全局统计同口径（s[s>-1]）仅对有效值求均值，避免哨兵被当
    # 真实指标平均导致all行失真（如完美模型被无GT类拉成0甚至负值）；
    # 某列全无效时保持-1（N/A语义），逐类行的-1显示保持COCO惯例不变
    raw_stats = class_stats
    valid = raw_stats > -1
    class_stats = np.full(raw_stats.shape[1], -1.0)
    for j in range(raw_stats.shape[1]):
        if valid[:, j].any():
            class_stats[j] = np.mean(raw_stats[valid[:, j], j])
    coco_metric_table.insert(0,["all",*class_stats])
    coco_metric_table = tabulate(coco_metric_table,
                                 tablefmt="pipe",numalign="center",floatfmt=".4f",
                                 headers=["类别","mAP@0.5:0.95","mAP@0.5","mAP@0.75","mAP@0.95",
                                          "mAR@0.5:0.95","mAR@0.5","mAR@0.75","mAR@0.95"])
    return class_stats, coco_metric_table


def save_coco_json(json_path,image_infos,annotations,class_names):
    """
    这是保存COCO格式标注JSON的函数（gt与pred共用）
    Args:
        json_path: 保存路径
        image_infos: 图像信息列表
        annotations: 标注信息数组（gt或pred）
        class_names: 类别名称数组
    Returns:
    """
    with open(json_path, 'w+', encoding="utf-8") as f:
        json_dict = {}
        json_dict['images'] = image_infos
        json_dict["annotations"] = annotations
        json_dict["categories"] = [{"id": id, "name": cls_name} for id, cls_name in enumerate(class_names)]
        json_data = json.dumps(json_dict, indent=4, separators=(',', ': '),
                               cls=NpEncoder, ensure_ascii=False)
        f.write(json_data)