# -*- coding: utf-8 -*-
# @Time    : 2025/3/21 14:23
# @Author  : DaiPuWei
# @Email   : puwei.dai@vitalchem.com
# @File    : prelabel_voc_dataset.py
# @Software: PyCharm

"""
    这是对voc数据集进行预标注的脚本
"""

import os
import cv2
import numpy as np

from tqdm import tqdm
from threading import Thread
from pascal_voc_writer import Writer

from model import build_model
from utils import NpEncoder
from utils import ArgsParser
from utils import init_config
from utils import init_logger

parser = ArgsParser()
parser.add_argument('--cfg', type=str, default='./config/detection.yaml', help='config yaml file path')
parser.add_argument('--dataset_dir', type=str, default='./voc_dataset', help='voc dataset directory')
parser.add_argument('--gpu_id', type=int, default=0, help='gpu id')
parser.add_argument('--num_threads', type=int, default=1, help='number of detection threads')
parser.add_argument('--print_detection_result', action='store_true', help='export time')
opt = parser.parse_args()

def prelabel_voc_dataset(logger,detection_models,voc_dataset_dir,print_detection_result=False):
    """
    这是利用检测模型对VOC数据集进行预标注的函数
    Args:
        logger: 日志类实例
        detection_models: 检测模型实例数组
        voc_dataset_dir: labelme数据集路径
        interval: 视频抽帧间隔，默认为-1
        print_detection_result: 是否打印检测结果，默认为False
    Returns:
    """
    # 初始化相关变量
    voc_dataset_dirs = []
    #print(source)
    logger.info("开始初始化视频文件")
    voc_annotation_dir = os.path.join(voc_dataset_dir,"Annotations")
    if os.path.exists(voc_annotation_dir):
        voc_dataset_dirs.append(voc_dataset_dir)
    else:
        for voc_dataset_name in os.listdir(voc_dataset_dir):
            _voc_dataset_dir = os.path.join(voc_dataset_dir,voc_dataset_name)
            voc_dataset_dirs.append(voc_dataset_dir)
    voc_dataset_dirs = np.array(voc_dataset_dirs)
    logger.info("结束初始化视频文件")
    logger.info("共有{}个voc数据集需要进行预标注".format(len(voc_dataset_dirs)))

    # 初始化图像路径和XML路径
    voc_image_paths = []
    voc_annotation_paths = []
    for voc_dataset_dir in voc_dataset_dirs:
        voc_image_dir = os.path.join(voc_dataset_dir,"JPEGImages")
        voc_annotation_dir = os.path.join(voc_dataset_dir,"Annotations")
        if not os.path.exists(voc_image_dir):
            voc_image_dir = os.path.join(voc_dataset_dir, "images")

        for image_name in os.listdir(voc_image_dir):
            fname,ext = os.path.splitext(image_name)
            voc_image_paths.append(os.path.join(voc_image_dir,image_name))
            voc_annotation_paths.append(os.path.join(voc_annotation_dir,fname+".xml"))
    voc_image_paths = np.array(voc_image_paths)
    voc_annotation_paths = np.array(voc_annotation_paths)
    logger.info("解析得到图片{0}张".format(len(voc_annotation_paths)))

    # 检测图像并生成VOC数据集标签
    logger.info("图像检测与预标注开始")
    prelabel_imageset_save_voc_dataset(detection_models,voc_image_paths,
                                       voc_annotation_paths,print_detection_result)
    logger.info("图像检测与预标注结束")

def prelabel_imageset_save_voc_dataset(detection_models,voc_image_paths,
                                       voc_annotation_paths,print_detection_result=False):
    """
    这是检测图像集进行预标注并保存为voc数据集的函数
    Args:
        detection_model: 检测模型实例猎豹
        voc_image_paths: voc数据集图像文件路径数组
        voc_annotation_paths: voc数据集标签文件路径数组
        print_detection_result: 是否打印检测结果，默认为False
    Returns:
    """
    # 多线程检测图像并生成VOC标签
    size = len(voc_image_paths)
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
        batch_voc_image_paths = voc_image_paths[start:end]
        batch_voc_annotation_paths = voc_annotation_paths[start:end]
        detection_model = detection_models[i]
        start = end
        t = Thread(target=detect_batch_images,
                   args=(detection_model,batch_voc_image_paths,
                         batch_voc_annotation_paths,print_detection_result))
        t.start()
        threads.append(t)
    for t in threads:
        t.join()

def detect_batch_images(detection_model,batch_voc_image_paths,
                        batch_voc_annotation_paths,print_detection_result=False):
    """
    这是利用检测模型检测批量图像并生成labelme标签的函数
    Args:
        detection_model: 检测模型
        batch_voc_image_paths: 批量VOC图像文件路径数组
        batch_voc_annotation_paths: 批量VOC标签文件路径数组
        print_detection_result: 是否打印检测结果，默认为False
    Returns:
    """
    for i in tqdm(np.arange(len(batch_voc_image_paths))):
        detect_single_image(detection_model,batch_voc_image_paths[i],
                            batch_voc_annotation_paths[i],print_detection_result)

def detect_single_image(detection_model,voc_image_path,voc_annotation_path,print_detection_result=False):
    """
    这是利用检测模型检测单张图像并保存VOC数据标签的函数
    Args:
        detection_model: 检测模型实例
        voc_image_path: voc图像文件路径
        voc_annotation_path: voc标签文件路径
        print_detection_result: 是否打印检测结果，默认为False
    Returns:
    """
    # 检测图像并将检测结果
    _,image_name = os.path.split(voc_image_path)
    image = cv2.imread(voc_image_path)
    h, w, _ = np.shape(image)
    outputs = detection_model.detect(image,export_time=False,print_detection_result=print_detection_result)

    # 将检测结果写入xml
    writer = Writer(voc_image_path,w,h)
    for output in outputs:
        x1,y1,x2,y2 = output['bbox']
        label = output['cls_name']
        writer.addObject(label,x1,y1,x2,y2)
    writer.save(voc_annotation_path)\

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
    prelabel_voc_dataset(logger, detection_models, opt.dataset_dir,opt.print_detection_result)

if __name__ == '__main__':
    run_main()
