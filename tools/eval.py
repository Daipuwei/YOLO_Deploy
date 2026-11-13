#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2026/9/11 08:07
# @Author  : DaiPuWei
# @Email   : daipuwei@qq.com
# @File    : eval.py
# @Software: PyCharm

"""
    这是在数据集上测试模型性能的脚本，支持VOC、COCO和YOLO数据集格式。

    评估工具函数（summarize_eval/eval_coco/eval_per_class/save_coco_json等）
    集中于utils/metric_utils/包，本脚本负责数据集遍历与结果累积编排。
"""

import os
import sys
import cv2
import numpy as np
from tqdm import tqdm
from datetime import datetime

# 启动路径优先级：虚拟环境已安装的库 > 源码树兜底。
# 库已安装时，从任意目录运行本脚本均可直接使用（不修改sys.path，
# 避免源码树遮蔽已安装版本）；未安装（源码直跑）时才把仓库根加入sys.path
try:
    import detection_track_toolkit  # noqa: F401  # 仅探测库是否已安装
except ImportError:
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from detection_track_toolkit.detection_model import build_model
from detection_track_toolkit.detection_model import draw_detection_results
from detection_track_toolkit.detection_model import x1y1x2y2_to_x1y1wh
from detection_track_toolkit.utils import ArgsParser
from detection_track_toolkit.utils import save_yaml
from detection_track_toolkit.utils import init_config
from detection_track_toolkit.utils import init_logger
from detection_track_toolkit.utils import build_dataset
from detection_track_toolkit.utils import save_coco_json
from detection_track_toolkit.utils import eval_coco
from detection_track_toolkit.utils import eval_per_class
from detection_track_toolkit.utils import compute_yolo_metrics

def convert_bbox_xyxy2xywh(annotation):
    """
    这是将标注bbox由x1y1x2y2转换为xywh的函数（COCO口径标注输入格式）
    Args:
        annotation: 标注字典（含bbox键）
    Returns:
        转换后标注字典（bbox键为xywh）
    """
    x1, y1, x2, y2 = annotation["bbox"]
    return {**annotation, "bbox": [x1, y1, x2 - x1, y2 - y1]}

def test(logger,detection_model,dataset_dir,result_dir,dataset_type='voc',choice='val',
         save_image=False,export_time=False,print_detection_result=False,metrics='coco,yolo'):
    """
    这是对模型性能进行测试的函数。
    检测循环结束后对“模型类别表中无GT标注的类别”输出warning
    （此类类别COCO逐类指标显示为N/A(-1)、YOLO口径整体指标不计入，
    提前告警避免误读result.txt中的指标）
    Args:
        logger: 日志类实例
        detection_model: 检测模型实例
        dataset_dir: 数据集地址
        result_dir: 结果文件夹地址
        dataset_type: 数据集类型，默认为'voc',候选值为['voc','coco','yolo_v1','yolo_v2']
        choice: 数据集子集类型，默认为'val'（单个子集字符串或子集字符串列表）
        save_image: 是否保存检测结果图像标志量，默认为False
        export_time: 是否输出模型推理时间，默认为False
        print_detection_result: 是否打印检测结果，默认为False
        metrics: 评估指标集合，'coco'与/或'yolo'，默认为'coco,yolo'
    Returns:
    """
    # 解析并校验评估指标集合（逗号分隔、大小写不敏感；未知指标直接报错，
    # 避免拼写错误静默跳过评估）
    metrics = str(metrics).lower().split(",")
    metrics = [m.strip() for m in metrics if m.strip()]
    if len(metrics) == 0:
        raise ValueError("评估指标不能为空，候选值为['coco','yolo']")
    for m in metrics:
        if m not in ("coco", "yolo"):
            raise ValueError("未知的评估指标'{0}'，候选值为['coco','yolo']".format(m))

    # choice支持单个子集字符串或子集字符串列表（统一规范为列表）
    choice = [choice] if isinstance(choice, str) else list(choice)

    # 初始化相关文件路径
    logger.info("开始加载检测数据集")
    batch_size = detection_model.get_batch_size()
    test_dataloader = build_dataset(dataset_type,dataset_dir,batch_size,choice)
    image_num = test_dataloader.get_image_num()
    logger.info("结束加载检测数据集")
    logger.info("共计有{}张图像需要进行检测".format(image_num))

    # 初始化json文件路径
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    result_txt_path = os.path.join(result_dir, 'result.txt')
    pred_json_result_path = os.path.join(result_dir, 'detection_result.json')
    gt_json_result_path = os.path.join(result_dir, 'gt_result.json')
    detect_image_dir = os.path.join(result_dir, 'images')
    if not os.path.exists(detect_image_dir):
        os.makedirs(detect_image_dir)
    detection_results = []
    image_infos = []
    gt_results = []

    # 检测图像
    detection_model_time_dict = {"preprocess_time": [],
                                 "inference_time": [],
                                 "postprocess_time": [],
                                 "detect_time": []}
    class_names = detection_model.get_class_names()
    colors = detection_model.get_colors()
    image_cnt = 0
    # 标注id从1开始计数：COCO标注id必须非零（COCOeval将dtMatches中的0
    # 视为未匹配，gt id为0会导致匹配成功的框被记为未匹配）；gt与pred
    # 的id空间相互独立，均可从1开始编号
    gt_annotation_cnt = 1
    pred_annotation_cnt = 1
    for batch_images,batch_meta_data in tqdm(test_dataloader):
        pred_results = detection_model.detect(batch_images,export_time,print_detection_result)
        if export_time:
            batch_preds, batch_time_dict = pred_results
        else:
            batch_preds = pred_results
            batch_time_dict = None
        # detect()在image_num==1时返回扁平列表（base_model降维），此处按
        # 本批实际图像数判断是否重新包裹为批量形态（修复：曾按模型batch_size
        # 判断，batch_size>1且尾批恰1张时不解包裹，zip把图像与检测字典错配，
        # 有框时pred["bbox"]崩溃/无框时该图像被静默丢弃致指标失真）
        if len(batch_images) == 1:
            batch_preds = [batch_preds]
        # 收集耗时（export_time为True时逐批记录，任意batchsize均有效）
        if batch_time_dict is not None:
            detection_model_time_dict['preprocess_time'].append(batch_time_dict["preprocess_time"])
            detection_model_time_dict['inference_time'].append(batch_time_dict["inference_time"])
            detection_model_time_dict['postprocess_time'].append(batch_time_dict["postprocess_time"])
            detection_model_time_dict['detect_time'].append(batch_time_dict["detect_time"])
        # 遍历每张图像的检测结果，并评估性能
        for image, meta_data,preds in zip(batch_images,batch_meta_data,batch_preds):
            image_path = meta_data['image_path']
            gts = meta_data['gt']
            _, image_name = os.path.split(image_path)
            # 保存检测图片
            if save_image:
                draw_image = draw_detection_results(image, preds,colors)
                cv2.imwrite(os.path.join(detect_image_dir, image_name), draw_image)
            # 初始化图像信息
            h, w, _ = np.shape(image)
            image_infos.append({'file_name': image_name, 'id': image_cnt, 'width': w, 'height': h})
            # 数据集gt统一为x1y1x2y2格式（VOC/COCO/YOLO加载器内部已对齐），
            # 直接存入gt_results（COCO口径落盘前再转回xywh）
            for obj_name,x1,y1,x2,y2 in gts:
                if obj_name not in class_names:
                    raise ValueError("gt标注类别'{0}'不在模型类别集{1}中，"
                                     "请检查class_name_path与数据集".format(obj_name, class_names))
                bbox_w = x2 - x1
                bbox_h = y2 - y1
                gt_results.append({'image_id': image_cnt,
                                      'iscrowd': 0,
                                      "bbox": [x1,y1,x2,y2],
                                      'area': bbox_w*bbox_h,
                                      "category_id": class_names.index(obj_name),
                                      'id': gt_annotation_cnt})
                gt_annotation_cnt += 1
            # 预测框为x1y1x2y2（与数据集gt格式统一），直接存入detection_results
            for pred in preds:
                x1,y1,x2,y2 = pred["bbox"]
                cls_id = pred['cls_id']
                score = pred['score']
                bbox_w = x2 - x1
                bbox_h = y2 - y1
                detection_results.append({'image_id': image_cnt,
                                          'iscrowd': 0,
                                          'category_id': cls_id,
                                          "bbox": [x1, y1, x2, y2],
                                          'area': bbox_w * bbox_h,
                                          'id': pred_annotation_cnt,
                                          'score': score})
                pred_annotation_cnt += 1
            image_cnt += 1
    # 计算检测时间（仅export_time为True时收集了计时数据）
    timing_lines = []
    if len(detection_model_time_dict['preprocess_time']) > 0:
        detection_model_time_dict['preprocess_time'] = np.mean(detection_model_time_dict["preprocess_time"])
        detection_model_time_dict['inference_time'] = np.mean(detection_model_time_dict["inference_time"])
        detection_model_time_dict['postprocess_time'] = np.mean(detection_model_time_dict["postprocess_time"])
        detection_model_time_dict['detect_time'] = np.mean(detection_model_time_dict["detect_time"])
        logger.info("batchsize={0},平均预处理时间为：{1:.4f}ms".format(batch_size,detection_model_time_dict['preprocess_time']))
        logger.info("batchsize={0},平均推理时间为：{1:.4f}ms".format(batch_size,detection_model_time_dict['inference_time']))
        logger.info("batchsize={0},平均后处理时间为：{1:.4f}ms".format(batch_size,detection_model_time_dict['postprocess_time']))
        logger.info("batchsize={0},平均检测时间为：{1:.4f}ms".format(batch_size,detection_model_time_dict['detect_time']))
        timing_lines = [
            "batchsize={0},平均预处理时间为：{1:.4f}ms\n".format(batch_size,detection_model_time_dict['preprocess_time']),
            "batchsize={0},平均推理时间为：{1:.4f}ms\n".format(batch_size,detection_model_time_dict['inference_time']),
            "batchsize={0},平均后处理时间为：{1:.4f}ms\n".format(batch_size,detection_model_time_dict['postprocess_time']),
            "batchsize={0},平均检测时间为：{1:.4f}ms\n".format(batch_size,detection_model_time_dict['detect_time']),
        ]

    logger.info("测试集路径为：{0}".format(dataset_dir))
    logger.info("评估性能所用的子集有：{0}",choice)

    # 预防告警：模型类别表中无GT的类别（检测循环结束后GT已收集完整）：
    # COCO逐类指标对无GT类显示为N/A(-1)，all行均值已剔除这些哨兵值；
    # YOLO口径整体指标只按有GT类别求均值
    gt_category_ids = set(gt["category_id"] for gt in gt_results)
    no_gt_classes = [class_names[i] for i in range(len(class_names))
                     if i not in gt_category_ids]
    if len(no_gt_classes) > 0:
        logger.warning("以下模型类别在测试集中无GT标注，逐类指标将显示为N/A：{0}"
                       .format(no_gt_classes))

    # 计算检测模型在测试集上的性能（COCOeval口径）
    if "coco" in metrics:
        # 保存gt与预测结果JSON（COCO口径标注输入为x1y1wh，落盘前统一转换）
        gt_coco_results = []
        for gt in gt_results:
            bbox = x1y1x2y2_to_x1y1wh(gt['bbox'])
            gt_coco_results.append({**gt,"bbox":bbox})
        detection_coco_results = []
        for pred in detection_results:
            bbox = x1y1x2y2_to_x1y1wh(pred['bbox'])
            detection_coco_results.append({**pred,"bbox":bbox})
        save_coco_json(gt_json_result_path, image_infos, gt_coco_results, class_names)
        logger.info("测试集真实标签JSON文件(COCO格式)保存路径为: {0}".format(gt_json_result_path))
        save_coco_json(pred_json_result_path, image_infos, detection_coco_results, class_names)
        logger.info("测试集预测结果JSON文件(COCO格式)保存路径为：{0}".format(pred_json_result_path))

        # 计算整体数据集的AP、AR指标
        evaluator, _, _, _ = eval_coco(gt_json_result_path, pred_json_result_path)
        # 计算每个类别的AP和AR指标
        class_stats, coco_metrics_table = eval_per_class(evaluator, class_names)

        # 打印模型在测试集上的检测性能
        logger.info("\n\n{0}\n{1}\n".format("COCO API下模型性能指标如下：",coco_metrics_table))

    # 计算Ultralytics风格YOLO口径指标（P/R/F1/mAP50/mAP50-95/混淆矩阵）
    if "yolo" in metrics:
        # 混淆矩阵按评测conf阈值统计，保证与P/R/mAP口径一致
        yolo_metrics, yolo_metrics_table, confusion_matrix_table = compute_yolo_metrics(
            detection_results, gt_results, class_names,
            conf_threshold=detection_model.confidence_threshold)
        logger.info("\n\n{0}\n{1}\n".format("Ultralytic YOLO API模型性能指标如下：",yolo_metrics_table))
        logger.info("\n\n{0}\n{1}\n".format("混淆矩阵如下（行=预测类别，列=真实类别，最后一行=漏检，最后一列=误检）：",confusion_matrix_table))

    # 保存结果到txt
    with open(result_txt_path, 'w+', encoding="utf-8") as f:
        f.writelines(timing_lines)
        f.write("\n\n\n")
        f.write("测试集路径为：{0}\n".format(dataset_dir))
        f.write("评估性能所用的子集有：{0}\n".format(choice))
        if "coco" in metrics:
            f.write("测试集真实标签JSON文件(COCO格式)保存路径为：{0}\n".format(gt_json_result_path))
            f.write("测试集预测结果JSON文件(COCO格式)保存路径为：{0}\n".format(pred_json_result_path))
            f.write("\n\n\n")
            f.write("COCO API下模型性能指标如下：\n")
            f.write(coco_metrics_table + "\n")
            f.write("\n\n\n")
        if "yolo" in metrics:
            f.write("Ultralytic YOLO API模型性能指标如下：\n")
            f.write(yolo_metrics_table + "\n")
            f.write("混淆矩阵如下（行=预测类别，列=真实类别，最后一行=漏检，最后一列=误检）：\n")
            f.write(confusion_matrix_table)

def run_main():
    """
    这是主函数
    """
    # 解析命令行参数,并初始化参数
    parser = ArgsParser()
    parser.add_argument('--cfg', type=str, default=os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), 'config', 'detection.yaml'), help='config yaml file path')
    parser.add_argument('--dataset_dir', type=str, default='./VOC2007', help='dataset directory')
    parser.add_argument('--dataset_type', type=str, default="voc", help='dataset type: voc, coco, yolo_v1 or yolo_v2')
    parser.add_argument('--choice', type=str, default="val",
                        help='dataset choice: single subset or comma-separated subsets (e.g. val or val,test)')
    parser.add_argument('--result_dir', type=str, default="./result/eval/", help='result save directory')
    parser.add_argument('--gpu_id', type=int, default=0, help='gpu id')
    parser.add_argument('--save_image', action='store_true', help='save detection image')
    parser.add_argument('--export_time', action='store_true', help='export detection time')
    parser.add_argument('--print_detection_result', action='store_true', help='print detection result')
    parser.add_argument('--metrics', type=str, default='coco,yolo', help='evaluation metrics: coco and/or yolo')
    opt = parser.parse_args()
    cfg = init_config(opt)

    # 初始化日志
    logger = init_logger(cfg["Log"])

    # 初始化检测模型
    detection_model = build_model(logger, cfg["DetectionModel"], gpu_id=opt.gpu_id)

    # 初始化相关路径路径
    time_str = datetime.now().strftime('%Y%m%d%H%M%S')
    dataset_dir = os.path.abspath(opt.dataset_dir)
    _,dataset_name = os.path.split(dataset_dir)
    model_path = cfg["DetectionModel"]["engine_model_path"]
    _,model_name = os.path.split(model_path)
    result_dir = os.path.join(opt.result_dir, dataset_name,model_name,time_str)
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    # 保存参数配置文件
    config_yaml_path = os.path.join(result_dir, "config.yaml")
    save_yaml(cfg, config_yaml_path)

    # 对检测模型进行预热，tensorRT等模型前几次推理时间较长，影响评测结果
    logger.info("预热模型开始")
    # 预热输入按模型输入布局构造（不再重复猜测布局启发式）
    if detection_model.is_nchw:
        image_tensor = np.random.random((detection_model.batch_size,detection_model.c,
                                         detection_model.h,detection_model.w))*255
    else:
        image_tensor = np.random.random((detection_model.batch_size,detection_model.h,
                                         detection_model.w,detection_model.c))*255
    image_tensor = image_tensor.astype(np.float32)
    # 预热直接调引擎推理：绕过预处理/后处理（detect()按HWC图像解释输入，
    # NCHW布局的随机张量走预处理会崩），只加热引擎本身
    for _ in range(100):
        detection_model.engine.inference([image_tensor])
    logger.info("预热模型结束")

    # choice支持单个子集字符串或逗号分隔的子集字符串列表（如"val,test"）
    choice = opt.choice
    if isinstance(choice, str) and "," in choice:
        choice = [c.strip() for c in choice.split(",") if c.strip()]

    # 测试检测性能
    test(logger,detection_model,dataset_dir,result_dir,
         opt.dataset_type,choice,opt.save_image,opt.export_time,
         opt.print_detection_result,opt.metrics)

    # 释放检测模型资源
    detection_model.release()

if __name__ == '__main__':
    run_main()
