# -*- coding: utf-8 -*-
# @Time    : 2025/3/20 15:56
# @Author  : DaiPuWei
# @Email   : daipuwei@qq.com
# @File    : prelabel_labelme_dataset.py
# @Software: PyCharm

"""
    这是对labelme数据集进行预标注的脚本
"""

import os
import cv2
import json
import numpy as np

from tqdm import tqdm
from threading import Thread

from model import build_model
from utils import NpEncoder
from utils import ArgsParser
from utils import init_config
from utils import init_logger

parser = ArgsParser()
parser.add_argument('--cfg', type=str, default='./config/detection.yaml', help='config yaml file path')
parser.add_argument('--dataset_dir', type=str, default='./labelme_dataset', help='labelme dataset directory')
parser.add_argument('--gpu_id', type=int, default=0, help='gpu id')
parser.add_argument('--num_threads', type=int, default=1, help='number of detection threads')
parser.add_argument('--print_detection_result', action='store_true', help='export time')
opt = parser.parse_args()

def prelabel_labelme_dataset(logger,detection_models,labelme_dataset_dir,print_detection_result=False):
    """
    这是利用检测模型对Labelme数据集进行预标注的函数
    Args:
        logger: 日志类实例
        detection_models: 检测模型实例数组
        labelme_dataset_dir: labelme数据集路径
        interval: 视频抽帧间隔，默认为-1
        print_detection_result: 是否打印检测结果，默认为False
    Returns:
    """
    # 初始化相关变量
    labelme_dataset_dirs = []
    #print(source)
    logger.info("开始初始化视频文件")
    labelme_image_dir = os.path.join(labelme_dataset_dir,"images")
    if os.path.exists(labelme_image_dir):
        labelme_dataset_dirs.append(labelme_image_dir)
    else:
        for labelme_dataset_name in os.listdir(labelme_dataset_dir):
            _labelme_dataset_dir = os.path.join(labelme_dataset_dir,labelme_dataset_name)
            labelme_dataset_dirs.append(_labelme_dataset_dir)
    labelme_dataset_dirs = np.array(labelme_dataset_dirs)
    logger.info("结束初始化视频文件")
    logger.info("共有{}个labelme数据集需要进行预标注".format(len(labelme_dataset_dirs)))

    # 初始化图像路径和XML路径
    labelme_image_paths = []
    labelme_json_paths = []
    for labelme_dataset_dir in labelme_dataset_dirs:
        labelme_image_dir = os.path.join(labelme_dataset_dir,"images")
        for image_name in os.listdir(labelme_image_dir):
            fname,ext = os.path.splitext(image_name)
            if ".json" in image_name:
                continue
            labelme_image_paths.append(os.path.join(labelme_image_dir,image_name))
            labelme_json_paths.append(os.path.join(labelme_image_dir,fname+".json"))
    labelme_image_paths = np.array(labelme_image_paths)
    labelme_json_paths = np.array(labelme_json_paths)
    logger.info("解析得到图片{0}张".format(len(labelme_image_paths)))

    # 检测图像并生成VOC数据集标签
    logger.info("图像检测与预标注开始")
    prelabel_imageset_save_labelme_dataset(detection_models,labelme_image_paths,
                                          labelme_json_paths,print_detection_result)
    logger.info("图像检测与预标注结束")

def prelabel_imageset_save_labelme_dataset(detection_models,labelme_image_paths,
                                          labelme_json_paths,print_detection_result=False):
    """
    这是检测图像集进行预标注并保存为labelme数据集的函数
    Args:
        detection_model: 检测模型实例猎豹
        labelme_image_paths: labelme数据集图像文件路径数组
        labelme_json_paths: labelme数据集标签文件路径数组
        print_detection_result: 是否打印检测结果，默认为False
    Returns:
    """
    # 多线程检测图像并生成VOC标签
    size = len(labelme_image_paths)
    num_models = len(detection_models)
    if size // num_models != 0:
        num_threads = num_models
    elif size // (num_models // 2) != 0:
        num_threads = num_models // 2
    elif size // (num_models // 4) != 0:
        num_threads = num_models // 4
    else:
        num_threads = 1
    batch_size = size // num_threads
    for i in np.arange(num_models - num_threads):
        del detection_models[0]
    start = 0
    threads = []
    for i in np.arange(num_threads):
        if i != num_threads-1:
            end = start + batch_size
        else:
            end = size
        batch_labelme_image_paths = labelme_image_paths[start:end]
        batch_labelme_json_paths = labelme_json_paths[start:end]
        detection_model = detection_models[i]
        start = end
        t = Thread(target=detect_batch_images,
                   args=(detection_model,batch_labelme_image_paths,
                         batch_labelme_json_paths,print_detection_result))
        t.start()
        threads.append(t)
    for t in threads:
        t.join()

def detect_batch_images(detection_model,batch_labelme_image_paths,
                        batch_labelme_json_paths,print_detection_result=False):
    """
    这是利用检测模型检测批量图像并生成labelme标签的函数
    Args:
        detection_model: 检测模型
        batch_labelme_image_paths: 批量VOC图像文件路径数组
        batch_labelme_json_paths: 批量VOC标签文件路径数组
        print_detection_result: 是否打印检测结果，默认为False
    Returns:
    """
    for i in tqdm(np.arange(len(batch_labelme_image_paths))):
        detect_single_image(detection_model,batch_labelme_image_paths[i],
                            batch_labelme_json_paths[i],print_detection_result)

def detect_single_image(detection_model,labelme_image_path,labelme_json_path,print_detection_result=False):
    """
    这是利用检测模型检测单张图像并保存VOC数据标签的函数
    Args:
        detection_model: 检测模型实例
        labelme_image_path: labelme图像文件路径
        labelme_json_path: labelme标签文件路径
        print_detection_result: 是否打印检测结果，默认为False
    Returns:
    """
    # 检测图像并将检测结果
    _,image_name = os.path.split(labelme_image_path)
    image = cv2.imread(labelme_image_path)
    h, w, _ = np.shape(image)
    outputs = detection_model.detect(image,export_time=False,print_detection_result=print_detection_result)

    # 将检测结果写入json
    json_data = {"version": "v0.0.1",
                 "flags": {},
                 "shapes": [],
                 "imagePath": image_name,
                 "imageData": None,
                 "imageHeight": h,
                 "imageWidth": w}
    for output in outputs:
        x1,y1,x2,y2 = output['bbox']
        label = output['cls_name']
        json_data['shapes'].append({
             "label": label,
             "points": [[x1, y1], [x2, y2]],
             "group_id": None,
             "shape_type": "rectangle",
             "flags": {},
        })

    # 将识别结果写入json
    with open(labelme_json_path, 'w', encoding='utf-8') as f:
        json_data = json.dumps(json_data, indent=4, cls=NpEncoder,
                               separators=(',', ': '), ensure_ascii=False)
        f.write(json_data)

def run_main():
    """
    这是主函数
    """
    # 初始化参数
    cfg = init_config(opt)
    logger = init_logger(cfg)

    # 初始化检测模型
    detection_models = []
    for i in np.arange(opt.num_threads):
        detection_model = build_model(logger, cfg["DetectionModel"], gpu_id=opt.gpu_id)
        detection_models.append(detection_model)

    # 初始化图像及其结果保存文件夹路径
    model_path = cfg["DetectionModel"]["engine_model_path"]
    _,model_name = os.path.split(model_path)

    # 检测图像
    prelabel_labelme_dataset(logger, detection_models, opt.dataset_dir,opt.print_detection_result)

if __name__ == '__main__':
    run_main()
