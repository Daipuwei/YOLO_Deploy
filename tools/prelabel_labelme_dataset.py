#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2026/9/11 02:00
# @Author  : DaiPuWei
# @Email   : daipuwei@qq.com
# @File    : prelabel_labelme_dataset.py
# @Software: PyCharm

"""
    这是对labelme数据集进行预标注的脚本（利用检测模型自动生成labelme标注json，
    供labelme打开后进行人工复核与修正）。

    模块内容：
        1. prelabel_labelme_dataset(logger, detection_models, ...)：预标注核心函数，
           完成labelme数据集路径初始化（扫描图像与标签json路径）与多线程分发执行；
        2. detect_batch_images(...)：单线程批量任务函数，按线程编号分行显示进度条；
        3. detect_single_image(...)：单张图像预标注函数，支持合并已有json标注
           （逐框IoU比对）与覆盖两种模式，单图异常隔离不中断线程；
        4. parse_labelme_json(labelme_json_path)：解析labelme标注json，
           获取每个目标的类别与定位（兼容历史遗留的2点与4点矩形points形式）；
        5. bbox_iou(box1, box2)：计算两个x1y1x2y2框的IoU。

    使用方式：
        python prelabel_labelme_dataset.py --cfg ./config/detection.yaml \
            --dataset_dir ./labelme_dataset --num_threads 4 \
            [--merge_existing_json --merge_iou_threshold 0.5] \
            [--skip_existing_json]
"""

import os
import sys
import cv2
import json
import numpy as np

from tqdm import tqdm
from threading import Thread, Lock

# 启动路径优先级：虚拟环境已安装的库 > 源码树兜底。
# 库已安装时，从任意目录运行本脚本均可直接使用（不修改sys.path，
# 避免源码树遮蔽已安装版本）；未安装（源码直跑）时才把仓库根加入sys.path
try:
    import detection_track_toolkit  # noqa: F401  # 仅探测库是否已安装
except ImportError:
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from detection_track_toolkit.detection_model import build_model
from detection_track_toolkit.utils import NpEncoder
from detection_track_toolkit.utils import ArgsParser
from detection_track_toolkit.utils import init_config
from detection_track_toolkit.utils import init_logger

# 图像文件后缀（labelme图像目录可能混存jpg/png/bmp等格式）
IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".JPG", ".JPEG", ".PNG", ".BMP")

def prelabel_labelme_dataset(logger,detection_models,labelme_dataset_dir,
                             print_detection_result=False,
                             merge_existing_json=False,merge_iou_threshold=0.7,
                             export_time=False,skip_existing_json=False):
    """
    这是利用检测模型对labelme数据集进行预标注的函数。
    核心流程：路径初始化（扫描图像与标签json路径）→ 线程数收敛 →
    多线程分发执行（图像分片、线程启动与汇合）。
    Args:
        logger: 日志类实例
        detection_models: 检测模型实例数组（每个线程独立使用一个模型实例，
            规避推理引擎非线程安全；多余模型保持空闲，由调用方统一释放）
        labelme_dataset_dir: labelme数据集路径（根目录含images子目录时直接使用，
            否则遍历一级子目录，仅保留含images子目录的目录）
        print_detection_result: 是否打印检测结果，默认为False
        merge_existing_json: 是否合并已有json标注，默认为False
            （True为合并模式，False为覆盖模式，语义见detect_single_image）
        merge_iou_threshold: 合并模式判定预测框与已有框重叠的IoU阈值，默认为0.7
        export_time: 是否导出检测时间，默认为False
        skip_existing_json: 是否跳过已存在json标注的图像，默认为False
            （True时图像旁已有同名json则跳过，用于断点续跑；json与图像同目录）
    Returns:
    """
    # 路径初始化：定位数据集images目录（根目录含images时直接使用，
    # 否则遍历一级子目录，仅保留含images子目录的目录，过滤文件与非数据集目录）
    labelme_dataset_dirs = []
    logger.info("开始初始化labelme数据集")
    labelme_image_dir = os.path.join(labelme_dataset_dir,"images")
    if os.path.exists(labelme_image_dir):
        # 列表元素统一为"数据集根目录"（含images子目录），
        # 与遍历一级子目录分支的元素形态一致（扫描循环会再拼images子路径）
        labelme_dataset_dirs.append(labelme_dataset_dir)
    else:
        for labelme_dataset_name in os.listdir(labelme_dataset_dir):
            _labelme_dataset_dir = os.path.join(labelme_dataset_dir,labelme_dataset_name)
            if os.path.isdir(_labelme_dataset_dir) and \
                    os.path.exists(os.path.join(_labelme_dataset_dir,"images")):
                labelme_dataset_dirs.append(_labelme_dataset_dir)
    if len(labelme_dataset_dirs) == 0:
        raise FileNotFoundError("未找到labelme数据集的images目录：{0}"
                                .format(labelme_dataset_dir))
    labelme_dataset_dirs = np.array(labelme_dataset_dirs)
    logger.info("共有{}个labelme数据集需要进行预标注".format(len(labelme_dataset_dirs)))

    # 路径初始化：扫描图像路径与对应标签json路径
    # （按图像后缀过滤，json等非图像文件不参与；json与图像同目录同名）
    labelme_image_paths = []
    labelme_json_paths = []
    skip_cnt = 0
    for labelme_dataset_dir in labelme_dataset_dirs:
        labelme_image_dir = os.path.join(labelme_dataset_dir,"images")
        for image_name in os.listdir(labelme_image_dir):
            if not image_name.endswith(IMAGE_EXTS):
                continue
            fname,_ = os.path.splitext(image_name)
            labelme_image_path = os.path.join(labelme_image_dir,image_name)
            labelme_json_path = os.path.join(labelme_image_dir,fname+".json")
            # 断点续跑：json与图像同目录同名，已存在则跳过该图像
            if skip_existing_json and os.path.exists(labelme_json_path):
                skip_cnt += 1
                continue
            labelme_image_paths.append(labelme_image_path)
            labelme_json_paths.append(labelme_json_path)
    labelme_image_paths = np.array(labelme_image_paths)
    labelme_json_paths = np.array(labelme_json_paths)
    if skip_existing_json and skip_cnt > 0:
        logger.info("跳过已存在json标注的图像{0}张".format(skip_cnt))
    logger.info("解析得到图片{0}张".format(len(labelme_image_paths)))
    size = len(labelme_image_paths)
    if size == 0:
        logger.info("无图像需要进行预标注")
        return

    # 检测模型实例数校验（传入本函数的强制判断，至少需要1个模型实例）
    num_models = len(detection_models)
    if num_models <= 0:
        raise ValueError("detection_models不能为空，至少需要1个检测模型实例")
    # 线程数收敛：不超过图像数与模型数，强制最小为1
    # （多余模型保持空闲，由调用方统一释放）
    num_threads = max(1, min(num_models, size))
    if num_threads < num_models:
        logger.info("图像数{0}少于模型数{1}，实际使用{2}个线程进行预标注"
                    .format(size, num_models, num_threads))

    # 多线程分发执行：图像分片（均分+尾批收尾），每线程独立模型与图像切片，
    # 各线程写各自的标签json文件，无共享写冲突
    logger.info("图像检测与预标注开始")
    # 打印检测结果互斥锁（避免多线程输出交织）
    print_lock = Lock()
    batch_size = size // num_threads
    start = 0
    threads = []
    for i in range(num_threads):
        if i != num_threads-1:
            end = start + batch_size
        else:
            end = size
        batch_labelme_image_paths = labelme_image_paths[start:end]
        batch_labelme_json_paths = labelme_json_paths[start:end]
        detection_model = detection_models[i]
        start = end
        t = Thread(target=detect_batch_images,
                   args=(logger,detection_model,batch_labelme_image_paths,
                         batch_labelme_json_paths,print_detection_result,
                         merge_existing_json,merge_iou_threshold,export_time,
                         i,print_lock))
        t.start()
        threads.append(t)
    for t in threads:
        t.join()
    logger.info("图像检测与预标注结束")

def detect_batch_images(logger,detection_model,batch_labelme_image_paths,
                        batch_labelme_json_paths,print_detection_result=False,
                        merge_existing_json=False,merge_iou_threshold=0.7,
                        export_time=False,thread_id=0,print_lock=None):
    """
    这是利用检测模型检测批量图像并生成labelme标签的函数（线程任务入口）
    Args:
        logger: 日志类实例
        detection_model: 检测模型实例（本线程独占使用）
        batch_labelme_image_paths: 批量labelme图像文件路径数组
        batch_labelme_json_paths: 批量labelme标签json文件路径数组
        print_detection_result: 是否打印检测结果，默认为False
        merge_existing_json: 是否合并已有json标注，默认为False
        merge_iou_threshold: 合并模式判定重叠的IoU阈值，默认为0.7
        export_time: 是否导出检测时间，默认为False
        thread_id: 线程编号（用于多线程进度条分行显示），默认为0
        print_lock: 打印检测结果互斥锁，默认为None
    Returns:
    """
    # position参数按线程编号分行显示进度条，避免多线程进度条互相覆盖
    for i in tqdm(range(len(batch_labelme_image_paths)), position=thread_id, leave=False):
        detect_single_image(logger,detection_model,batch_labelme_image_paths[i],
                            batch_labelme_json_paths[i],print_detection_result,
                            merge_existing_json,merge_iou_threshold,export_time,
                            print_lock)

def bbox_iou(box1,box2):
    """
    这是计算两个x1y1x2y2框IoU的函数（合并模式下判定预测框与已有框重叠程度）
    Args:
        box1: 框1坐标，shape为(4,)
        box2: 框2坐标，shape为(4,)
    Returns:
        IoU值（两框无交集或并集为0时返回0.0）
    """
    # 交集宽高（无交集时为0）
    x1, y1, x2, y2 = box1
    x1_, y1_, x2_, y2_ = box2
    inter_w = max(0.0, min(x2, x2_) - max(x1, x1_))
    inter_h = max(0.0, min(y2, y2_) - max(y1, y1_))
    inter = inter_w * inter_h
    if inter <= 0:
        return 0.0
    # 并集面积与IoU（并集为0时返回0.0兜底）
    area1 = (x2 - x1) * (y2 - y1)
    area2 = (x2_ - x1_) * (y2_ - y1_)
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0.0

def parse_labelme_json(labelme_json_path):
    """
    这是解析labelme数据集json标签文件，获取每个目标分类与定位的函数。
    【当前仅支持矩形框】只解析rectangle矩形标注（兼容历史遗留的2点与
    4点两种points形式：4点取对角线点、2点直接取两点）；polygon/point/
    line/circle等非矩形shape_type暂不支持——本工具集的检测目标即矩形框，
    且自身写出的json恒为4点矩形，故按矩形口径解析。
    后续如需支持更多标注类型：在此处按shape['shape_type']分流即可
    （rectangle走现有矩形解析；polygon等按各自几何语义转换或跳过告警），
    无需改动调用方。
    Args:
        labelme_json_path: labelme json标签文件路径
    Returns:
        objects: 目标列表，每个目标为[类别名,x1,y1,x2,y2]（像素坐标）
        (image_h, image_w): 图像高宽元组
    """
    objects = []
    try:
        with open(labelme_json_path, 'r', encoding='utf-8') as f:
            # 读取json文件中的检测标签
            json_data = json.load(f)
            image_h = json_data["imageHeight"]
            image_w = json_data["imageWidth"]
            for shape in json_data['shapes']:
                # 当前仅支持矩形框（见函数docstring）：仅按points点数区分
                # 2点/4点两种矩形形式解析，不读shape_type；非矩形标注
                # （polygon/point/line等）会被当矩形错读，属已知边界，
                # 后续支持更多类型时在此按shape_type分流
                bboxes = np.reshape(shape['points'], (-1, 2))
                if len(bboxes) == 4:
                    # 4点矩形：取对角线两点（约定第0点为左上、第2点为右下）
                    x1, y1 = bboxes[0]
                    x2, y2 = bboxes[2]
                else:
                    # 2点矩形：直接取两点（左上与右下）
                    x1, y1 = bboxes[0]
                    x2, y2 = bboxes[1]
                cls_name = shape['label']
                objects.append([cls_name, int(x1), int(y1), int(x2), int(y2)])
        return objects, (image_h, image_w)
    except UnicodeDecodeError:
        print("[ERROR] 编码有问题的文件: {0}".format(labelme_json_path))
        raise  # 重新抛出，由detect_single_image的异常隔离跳过该图像并记录日志

def detect_single_image(logger,detection_model,labelme_image_path,labelme_json_path,
                        print_detection_result=False,
                        merge_existing_json=False,merge_iou_threshold=0.7,
                        export_time=False,print_lock=None):
    """
    这是利用检测模型检测单张图像并保存labelme标签json的函数。
    两种模式均先解析一次已有json标注，再统一逐框比对预测框与已有框的IoU：
    - merge_existing_json为True（合并模式）：重叠程度不低于merge_iou_threshold
      的预测框从预测框列表中删除（保留已有json中的框标签），不重叠或低于
      阈值的预测框追加写入json；已有标注为空时预测框直接作为标注；
    - 为False（覆盖模式）：不管是否存在json文件，json中的框标签全部按预测框写入。
    输出shapes统一为4点矩形形式（左上角起顺时针）；单图失败时记录错误日志
    并跳过，不中断所在线程。
    Args:
        logger: 日志类实例
        detection_model: 检测模型实例
        labelme_image_path: labelme图像文件路径
        labelme_json_path: labelme标签json文件路径
        print_detection_result: 是否打印检测结果，默认为False
        merge_existing_json: 是否合并已有json标注，默认为False
        merge_iou_threshold: 合并模式判定预测框与已有框重叠的IoU阈值，默认为0.7
        export_time: 是否导出检测时间，默认为False
        print_lock: 打印检测结果互斥锁（避免多线程输出交织），默认为None
    Returns:
    """
    try:
        # 读取图像（读取失败抛出IOError，由本函数异常隔离统一处理）
        _,image_name = os.path.split(labelme_image_path)
        image = cv2.imread(labelme_image_path)
        if image is None:
            raise IOError("图像读取失败：{0}".format(labelme_image_path))
        h, w, _ = np.shape(image)
        # 打印检测结果时加锁串行输出，避免多线程输出交织
        if print_detection_result and print_lock is not None:
            with print_lock:
                outputs = detection_model.detect(image,export_time=export_time,
                                                 print_detection_result=print_detection_result)
        else:
            outputs = detection_model.detect(image,export_time=export_time,
                                             print_detection_result=print_detection_result)
        # export_time为True时detect返回(检测结果,时间信息字典)，拆包取检测结果
        if export_time:
            outputs = outputs[0]

        # 解析已有json标注（两种模式统一解析一次，不存在json时为空）
        existing_objects = []
        if os.path.exists(labelme_json_path):
            existing_objects, _ = parse_labelme_json(labelme_json_path)

        # 预测框转为与已有标注同构的[类别名,x1,y1,x2,y2]列表
        pred_objects = []
        for output in outputs:
            x1,y1,x2,y2 = output['bbox']
            pred_objects.append([output['cls_name'], x1, y1, x2, y2])

        # 覆盖模式丢弃已有标注，或已有标注为空：预测框直接作为标注
        if not merge_existing_json or len(existing_objects) == 0:
            objects = pred_objects
        else:
            # 合并模式：双重遍历，预测框与任一【同类别】已有框重叠达到
            # 阈值则从预测框列表中删除（保留已有框标签）；
            # 遍历结束后若预测框不为空，将剩余预测框加入已有标注。
            # 修复：曾只比坐标IoU不比类别，不同类别但空间高度重叠的对象
            # （如"人"骑"自行车"）预测框被误判重复而删除
            existing_boxes = [obj[1:5] for obj in existing_objects]
            existing_labels = [obj[0] for obj in existing_objects]
            remaining_preds = []
            for pred in pred_objects:
                pred_label, pred_box = pred[0], pred[1:5]
                keep = True
                for label, box in zip(existing_labels, existing_boxes):
                    if pred_label == label and \
                            bbox_iou(pred_box, box) >= merge_iou_threshold:
                        keep = False
                        break
                if keep:
                    remaining_preds.append(pred)
            if len(remaining_preds) > 0:
                existing_objects.extend(remaining_preds)
            objects = existing_objects

        # 统一序列化：4点矩形形式（左上角起顺时针），
        # 历史遗留的2点标注经此统一为4点形式写入
        shapes = []
        for obj in objects:
            label, x1, y1, x2, y2 = obj
            shapes.append({"label": label,
                           "points": [[x1, y1], [x2, y1], [x2, y2], [x1, y2]],
                           "group_id": None,
                           "shape_type": "rectangle",
                           "flags": {}})

        # 构造labelme json数据（imageData置None，由labelme按imagePath读取图像）
        json_data = {"version": "v0.0.1",
                     "flags": {},
                     "shapes": shapes,
                     "imagePath": image_name,
                     "imageData": None,
                     "imageHeight": h,
                     "imageWidth": w}

        # 将标注结果写入json文件
        with open(labelme_json_path, 'w', encoding='utf-8') as f:
            json_data = json.dumps(json_data, indent=4, cls=NpEncoder,
                                   separators=(',', ': '), ensure_ascii=False)
            f.write(json_data)
    except (RuntimeError, MemoryError, KeyError, TypeError, AttributeError,
            IndexError, PermissionError) as e:
        # 系统级错误（模型/引擎推理失败、OOM、json缺必需字段/契约错、
        # 磁盘写权限等）：原样重抛，避免整批预标注白跑却以退出码0
        # 静默失败的"假成功"（修复：曾一并吞掉，N张图只打N条error、
        # 退出码0、0产出，命令行无法察觉）
        raise
    except Exception as e:
        # 单图级错误（读图失败/图像解码异常等偶发）：跳过该图并记录日志，
        # 不中断整个线程（单图隔离，一张坏图不拖垮整个批次）
        logger.error("预标注失败，跳过图像：{0}：{1}".format(labelme_image_path, e))

def run_main():
    """
    这是主函数（参数解析与传递：非法参数修正、logger与检测模型初始化、
    预标注调度与模型资源释放在此完成）
    """
    # 解析命令行参数，并初始化参数
    parser = ArgsParser()
    parser.add_argument('--cfg', type=str, default=os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), 'config', 'detection.yaml'), help='config yaml file path')
    parser.add_argument('--dataset_dir', type=str, default='./labelme_dataset', help='labelme dataset directory')
    parser.add_argument('--gpu_id', type=int, default=0, help='gpu id')
    parser.add_argument('--num_threads', type=int, default=1, help='number of detection threads')
    parser.add_argument('--print_detection_result', action='store_true', help='print detection result')
    parser.add_argument('--export_time', action='store_true', help='export detection time')
    parser.add_argument('--merge_existing_json', action='store_true',
                        help='merge with existing json annotations: keep existing boxes whose IoU '
                             'with predictions reaches threshold, append non-overlapping predictions; '
                             'if False, overwrite all boxes with predictions')
    parser.add_argument('--merge_iou_threshold', type=float, default=0.7,
                        help='IoU threshold (0-1) for judging overlap in merge_existing_json mode')
    parser.add_argument('--skip_existing_json', action='store_true',
                        help='skip images whose co-located json annotation already exists '
                             '(json and image share the same directory)')
    opt = parser.parse_args()
    cfg = init_config(opt)

    # 初始化日志
    logger = init_logger(cfg["Log"])

    # 参数修正：线程数非法（<=0）时直接置为1，
    if opt.num_threads <= 0:
        logger.warning("num_threads非法（{0}），已置为1".format(opt.num_threads))
        opt.num_threads = 1
    # 合并IoU阈值超出[0,1]区间时直接置为0.7
    if not 0.0 <= opt.merge_iou_threshold <= 1.0:
        logger.warning("merge_iou_threshold非法（{0}），已置为0.7".format(opt.merge_iou_threshold))
        opt.merge_iou_threshold = 0.7

    # 初始化检测模型（每个线程独立模型实例，规避推理引擎非线程安全）
    detection_models = []
    for i in range(opt.num_threads):
        detection_model = build_model(logger, cfg["DetectionModel"], gpu_id=opt.gpu_id)
        detection_models.append(detection_model)

    # 预标注labelme数据集（路径初始化与多线程分发执行在prelabel_labelme_dataset中完成）
    prelabel_labelme_dataset(logger, detection_models, opt.dataset_dir,
                             print_detection_result=opt.print_detection_result,
                             merge_existing_json=opt.merge_existing_json,
                             merge_iou_threshold=opt.merge_iou_threshold,
                             export_time=opt.export_time,
                             skip_existing_json=opt.skip_existing_json)

    # 释放检测模型资源
    for detection_model in detection_models:
        detection_model.release()

if __name__ == '__main__':
    run_main()
