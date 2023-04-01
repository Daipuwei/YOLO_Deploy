# -*- coding: utf-8 -*-
# @Time    : 2023/4/1 下午11:00
# @Author  : DaiPuWei
# @Email   : daipuwei@qq.com
# @File    : imageset2voc_dataset.py
# @Software: PyCharm


"""
    这是将图像集转换为VOC数据集的脚本
"""

import os
import cv2
import argparse
import numpy as np
from tqdm import tqdm
from threading import Thread
from datetime import datetime
from pascal_voc_writer import Writer

from utils import load_yaml
from utils import logger_config

IMG_FORMATS = ['.bmp', '.dng', '.jpeg', '.jpg', '.mpo', '.png', '.tif', '.tiff', '.webp']  # include image suffixes

parser = argparse.ArgumentParser()
parser.add_argument('--cfg', type=str, default='./config/detection.yaml', help='config yaml file path')
parser.add_argument('--imageset', type=str, default='./imageset', help='imageset(s) directory')
parser.add_argument('--result_dir', type=str, default="./result", help='voc dataset save directory')
parser.add_argument('--num_threads', type=int, default=1, help='number of detection threads')
parser.add_argument('--confidence_threshold', type=float, default=0.5, help='detection confidence_threshold')
opt = parser.parse_args()

def imageset2voc_dataset(logger,detection_model,imageset,result_dir,num_threads=1,confidence_threshold=0.1):
    """
    这是利用检测模型对图像(集)进行预标注并生成VOC数据集的函数
    Args:
        logger: 日志类实例
        detection_model: 检测模型实例
        imageset: (批量)图像集文件夹
        result_dir: 结果保存路径
        interval: 视频抽帧间隔，默认为1s
        confidence_threshold: 检测结果置信度阈值,用于判定检测结果是否写入XML文件,默认为0.1
    Returns:
    """
    # 初始化视频路径
    time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    result_dir = os.path.join(result_dir,time)
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    # result_dir = os.path.abspath(result_dir)
    # if not os.path.exists(result_dir):
    #     os.makedirs(result_dir)

    # 初始化图像文件路径及其VOC相关文件路径
    logger.info("开始初始化图像文件")
    for file in os.listdir(imageset):
        print(file)
        if os.path.isfile(file):           # 是文件则为单个图像集
            fname, ext = os.path.splitext(file)
            if ext in IMG_FORMATS:
                is_single_imageset = True
                break
        else:
            is_single_imageset = False
            break
    image_paths = []
    voc_image_paths = []
    voc_xml_paths = []
    if is_single_imageset:                          # 单个图像数据集
        imageset_dirs = [imageset]
    else:                                           # 批量图像数据集
        imageset_dirs = []
        for imageset_name in os.listdir(imageset):
            imageset_dir = os.path.join(imageset, imageset_name)
            if os.path.isdir(imageset_dir):
                imageset_dirs.append(imageset_dir)
    num_imageset = len(imageset_dirs)
    for imageset_dir in imageset_dirs:
        _, imageset_name = os.path.split(imageset_dir)
        voc_dataset_dir = os.path.join(result_dir, imageset_name)
        voc_image_dir = os.path.join(voc_dataset_dir, "JPEGImages")
        voc_annotation_dir = os.path.join(voc_dataset_dir, "Annotations")
        if not os.path.exists(voc_image_dir):
            os.makedirs(voc_image_dir)
        if not os.path.exists(voc_annotation_dir):
            os.makedirs(voc_annotation_dir)
        for i, image_name in enumerate(os.listdir(imageset_dir)):
            image_paths.append(os.path.join(imageset_dir, image_name))
            voc_image_paths.append(os.path.join(voc_image_dir, "{0}_frame{1:08d}.jpg".format(imageset_name, i)))
            voc_xml_paths.append(os.path.join(voc_annotation_dir, "{0}_frame{1:08d}.xml".format(imageset_name, i)))
    image_paths = np.array(image_paths)
    voc_image_paths = np.array(voc_image_paths)
    voc_xml_paths = np.array(voc_xml_paths)
    logger.info("结束初始化图像文件")
    logger.info("共有{}个图像集需要转换成VOC数据集，总计图像{}张".format(num_imageset,len(image_paths)))

    # 检测图像并生成VOC数据集标签
    logger.info("图像检测与预标注开始")
    detect_imageset_save_voc_annotations(detection_model,image_paths,voc_image_paths,
                                         voc_xml_paths,num_threads,confidence_threshold)
    logger.info("图像检测与预标注结束")


def detect_imageset_save_voc_annotations(detection_model,image_paths,
                                         voc_image_paths,voc_xml_paths,num_threads=1,confidence_threshold=0.1):
    """
    这是检测图像集并保存VOC标签集的函数
    Args:
        detection_model: 检测模型实例
        image_paths: 图像路径数组
        voc_image_paths: VOC图像文件路径数组
        voc_xml_paths: VOC标签文件路径数组
        num_threads: 检测线程数,默认为1
        confidence_threshold: 检测结果置信度阈值,用于判定检测结果是否写入XML文件,默认为0.1
    Returns:
    """
    # 多线程检测图像并生成VOC标签
    size = len(voc_image_paths)
    batch_size = size // num_threads
    start = 0
    threads = []
    for i in np.arange(num_threads):
        if i != num_threads-1:
            end = start + batch_size
        else:
            end = size
        batch_image_paths = image_paths[start:end]
        batch_voc_image_paths = voc_image_paths[start:end]
        batch_voc_xml_paths = voc_xml_paths[start:end]
        _detection_model = detection_model[i]
        start = end
        t = Thread(target=detect_batch_images_save_voc_annotations,
                   args=(_detection_model,batch_image_paths,
                         batch_voc_image_paths,batch_voc_xml_paths,confidence_threshold))
        t.start()
        threads.append(t)
    for t in threads:
        t.join()


def detect_batch_images_save_voc_annotations(detection_model,batch_image_paths,
                                             batch_voc_image_paths,batch_voc_xml_paths,confidence_threshold=0.1):
    """
    这是利用检测模型检测批量图像并生成VOC标签的函数
    Args:
        detection_model: 检测模型
        batch_image_paths: 批量文件路径数组
        batch_voc_image_paths: 批量VOC图像文件路径数组
        batch_voc_xml_paths: 批量VOC标签文件路径数组
        confidence_threshold: 检测结果置信度阈值,用于判定检测结果是否写入XML文件,默认为0.1
    Returns:
    """
    for i in tqdm(np.arange(len(batch_voc_image_paths))):
        detect_single_image_save_voc_annotation(detection_model,batch_image_paths[i],
                                                batch_voc_image_paths[i],batch_voc_xml_paths[i],confidence_threshold)

def detect_single_image_save_voc_annotation(detection_model,image_path,voc_image_path,voc_xml_path,confidence_threshold=0.1):
    """
    这是利用检测模型检测单张图像并保存VOC数据标签的函数
    Args:
        detection_model: 检测模型实例
        image_path: 图像文件路径
        voc_image_path: VOC图像文件路径
        voc_xml_path: VOC标签文件路径
        confidence_threshold: 检测结果置信度阈值,用于判定检测结果是否写入XML文件,默认为0.1
    Returns:
    """
    # 初始化VOC标签写入类
    image = cv2.imread(image_path)
    h, w, c = np.shape(image)
    writer = Writer(voc_image_path,w,h)

    # 复制图像
    cv2.imwrite(voc_image_path,image)

    # 检测图像并将检测结果写入XML
    class_names = detection_model.get_class_names()
    preds = detection_model.detect(image)
    if len(preds) > 0:
        for x1, y1, x2, y2, score, cls_id in preds:
            if score > confidence_threshold:
                x1 = int(round(x1))
                y1 = int(round(y1))
                x2 = int(round(x2))
                y2 = int(round(y2))
                cls_id = int(cls_id)
                writer.addObject(class_names[cls_id], x1, y1, x2, y2)
        writer.save(voc_xml_path)

def run_main():
    """
    这是主函数
    """
    # 初始化检测模型
    cfg = load_yaml(opt.cfg)
    model_type = cfg["DetectionModel"]["model_type"].lower()
    logger = logger_config(cfg['log_path'], model_type)
    detection_models = []
    for i in np.arange(opt.num_threads):
        if model_type == 'yolov5':
            from model import YOLOv5
            detection_model = YOLOv5(logger=logger, cfg=cfg)
        else:
            from model import YOLOv5
            detection_model = YOLOv5(logger=logger, cfg=cfg)
        detection_models.append(detection_model)

    # 初始化图像及其结果保存文件夹路径
    # time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    # result_dir = os.path.join(opt.result_dir, time)
    result_dir = os.path.abspath(opt.result_dir)
    imageset = os.path.abspath(opt.imageset)

    # 检测图像
    imageset2voc_dataset(logger, detection_models, imageset,result_dir,opt.num_threads,opt.confidence_threshold)

if __name__ == '__main__':
    run_main()
