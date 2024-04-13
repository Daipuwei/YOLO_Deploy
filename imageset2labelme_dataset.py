# -*- coding: utf-8 -*-
# @Time    : 2024/1/15 下午10:34
# @Author  : DaiPuWei
# @Email   : daipuwei@qq.com
# @File    : imageset2labelme_dataset.py
# @Software: PyCharm

"""
   这是将图像集利用检测模型进行预标注转换为Labelme数据集的脚本
"""

import os
import sys
import cv2
import json
import shutil
import labelme
import numpy as np

from tqdm import tqdm
from threading import Thread
from datetime import datetime

from model import build_model
from utils import NpEncoder
from utils import ArgsParser
from utils import init_config
from utils import logger_config

IMG_FORMATS = ['.bmp', '.dng', '.jpeg', '.jpg', '.mpo', '.png', '.tif', '.tiff', '.webp']  # include image suffixes

parser = ArgsParser()
parser.add_argument('--cfg', type=str, default='./config/detection.yaml', help='config yaml file path')
parser.add_argument('--imageset', type=str, default='./imageset', help='imageset(s) directory')
parser.add_argument('--result_dir', type=str, default="./result", help='voc dataset save directory')
parser.add_argument('--num_threads', type=int, default=1, help='number of detection threads')
parser.add_argument('--print_detection_result', action='store_true', help='export time')
opt = parser.parse_args()

def imageset2labelme_dataset(logger,detection_models,imageset,result_dir,print_detection_result=False):
    """
    这是利用检测模型对图像(集)进行预标注并生成Labelme数据集的函数
    Args:
        logger: 日志类实例
        detection_models: 检测模型实例数组
        imageset: (批量)图像集文件夹
        result_dir: 结果保存路径
        print_detection_result: 是否打印检测结果，默认为False
    Returns:
    """
    result_dir = os.path.abspath(result_dir)
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    # 初始化图像文件路径及其VOC相关文件路径
    logger.info("开始初始化图像文件")
    for file in os.listdir(imageset):
        file_path = os.path.join(imageset,file)
        if os.path.isfile(file_path):           # 是文件则为单个图像集
            _, ext = os.path.splitext(file)
            if ext in IMG_FORMATS:
                is_single_imageset = True
                break
        else:
            is_single_imageset = False
            break
    # 初始化相关路径
    image_paths = []
    labelme_image_paths = []
    labelme_json_paths = []
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
        labelme_dataset_dir = os.path.join(result_dir, imageset_name)
        labelme_image_dir = os.path.join(labelme_dataset_dir, "images")
        if not os.path.exists(labelme_image_dir):
            os.makedirs(labelme_image_dir)
        for i, image_name in enumerate(os.listdir(imageset_dir)):
            image_paths.append(os.path.join(imageset_dir, image_name))
            labelme_image_paths.append(os.path.join(labelme_image_dir,
                                                    "{0}_frame{1:08d}.jpg".format(imageset_name, i)))
            labelme_json_paths.append(os.path.join(labelme_image_dir,
                                                   "{0}_frame{1:08d}.json".format(imageset_name, i)))
    image_paths = np.array(image_paths)
    labelme_image_paths = np.array(labelme_image_paths)
    labelme_json_paths = np.array(labelme_json_paths)
    logger.info("结束初始化图像文件")
    logger.info("共有{}个图像集需要转换成Labelme数据集，总计图像{}张".format(num_imageset,len(image_paths)))

    # 检测图像进行预标注并生成Labelme数据集标签
    logger.info("图像检测与预标注开始")
    prelabel_imageset_save_labelme_dataset(detection_models,image_paths,
                                           labelme_image_paths,labelme_json_paths,print_detection_result)
    logger.info("图像检测与预标注结束")


def prelabel_imageset_save_labelme_dataset(detection_models,image_paths,
                                           labelme_image_paths,labelme_json_paths,print_detection_result=False):
    """
    这是检测图像集进行预标注并保存为labelme数据集格式的函数
    Args:
        detection_models: 检测模型实例数组
        image_paths: 图像路径数组
        labelme_image_paths: Labelme数据集图像文件路径数组
        labelme_json_paths: Labelme数据集标签文件路径数组
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
    for i in np.arange(num_models-num_threads):
        del detection_models[0]
    start = 0
    threads = []
    for i in np.arange(num_threads):
        if i != num_threads-1:
            end = start + batch_size
        else:
            end = size
        batch_image_paths = image_paths[start:end]
        batch_labelme_image_paths = labelme_image_paths[start:end]
        batch_labelme_json_paths = labelme_json_paths[start:end]
        detection_model = detection_models[i]
        start = end
        t = Thread(target=detect_batch_images,
                   args=(detection_model,batch_image_paths,
                         batch_labelme_image_paths,batch_labelme_json_paths,print_detection_result))
        t.start()
        threads.append(t)
    for t in threads:
        t.join()

def detect_batch_images(detection_model,batch_image_paths,
                        batch_labelme_image_paths,batch_labelme_json_paths,print_detection_result=False):
    """
    这是利用检测模型检测批量图像进行预标注并保存为Labelme数据集格式的函数
    Args:
        detection_model: 检测模型
        batch_image_paths: 批量文件路径数组
        batch_labelme_image_paths: 批量Labelme数据集图像文件路径数组
        batch_labelme_json_paths: 批量Labelme数据集标签文件路径数组
        print_detection_result: 是否打印检测结果，默认为False
    Returns:
    """
    for i in tqdm(np.arange(len(batch_labelme_image_paths))):
        detect_single_image(detection_model,batch_image_paths[i],
                            batch_labelme_image_paths[i],batch_labelme_json_paths[i],print_detection_result)

def detect_single_image(detection_model,image_path,
                        labelme_image_path,labelme_json_path,print_detection_result=False):
    """
    这是利用检测模型检测单张图像进行预标注并保存为Labelme数据集格式的函数
    Args:
        detection_model: 检测模型实例
        image_path: 图像文件路径
        labelme_image_path: labelme数据集图像文件路径
        labelme_json_path: labelme数据集标签文件路径
        print_detection_result: 是否打印检测结果，默认为False
    Returns:
    """
    # 复制图像
    shutil.copy(image_path,labelme_image_path)

    # 检测图像并将检测结果
    _,image_name = os.path.split(image_path)
    image = cv2.imread(image_path)
    h, w, _ = np.shape(image)
    outputs = detection_model.detect(image,export_time=False,print_detection_result=print_detection_result)

    # 对检测结果进行编码
    json_data = {"version": labelme.__version__,
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

    # 写入json
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
    for _ in np.arange(opt.num_threads):
        detection_model = build_model(logger, cfg["DetectionModel"], gpu_id=opt.gpu_id)
        detection_models.append(detection_model)

    # 初始化图像及其结果保存文件夹路径
    time = datetime.now().strftime('%Y%m%d%H%M%S')
    model_path = cfg["DetectionModel"]["engine_model_path"]
    _,model_name = os.path.split(model_path)
    result_dir = os.path.join(opt.result_dir,model_name,time)
    imageset = os.path.abspath(opt.imageset)
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    # 检测图像
    imageset2labelme_dataset(logger, detection_models, imageset,result_dir,opt.num_threads)

if __name__ == '__main__':
    run_main()
