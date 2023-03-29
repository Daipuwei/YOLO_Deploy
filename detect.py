# -*- coding: utf-8 -*-
# @Time    : 2023/3/28 下午2:43
# @Author  : DaiPuWei
# @Email   : daipuwei@qq.com
# @File    : detect.py
# @Software: PyCharm

"""
    这是检测图像和视频的脚本
"""

import os
import cv2
import argparse
import numpy as np
from tqdm import tqdm
from datetime import datetime

from utils import load_yaml
from utils import logger_config
from utils import draw_detection_results

IMG_FORMATS = ['.bmp', '.dng', '.jpeg', '.jpg', '.mpo', '.png', '.tif', '.tiff', '.webp']  # include image suffixes
VID_FORMATS = ['.asf', '.avi', '.gif', '.m4v', '.mkv', '.mov', '.mp4', '.mpeg', '.mpg', '.ts', '.wmv']  # include video suffixes

parser = argparse.ArgumentParser()
parser.add_argument('--cfg', type=str, default='./config/detection.yaml', help='config yaml file path')
parser.add_argument('--source', type=str, default='./video', help='image path or image directory or video path or video directory')
parser.add_argument('--result_dir', type=str, default="./result", help='video detection result save directory')
parser.add_argument('--interval', type=int, default=1, help='video interval')
opt = parser.parse_args()

def detect(logger,detection_model,source,result_dir,interval=1):
    """
    这是利用检测模型检测图像或者视频的函数
    Args:
        logger: 日志类
        detection_model: 检测模型
        source: 图像文件或者图像文件夹或者视频文件路径或者视频文件夹
        result_dir: 结果保存路径
        interval: 视频抽帧间隔,默认为1s
    Returns:
    """
    # 初始化视频路径
    result_dir = os.path.abspath(result_dir)
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    video_paths = []
    image_paths = []
    image_result_paths = []
    #print(source)
    if os.path.isfile(source):               # 单个视频
        dir,filename = os.path.split(source)
        fname,ext = os.path.splitext(filename)
        if ext in IMG_FORMATS:
            image_paths.append(os.path.abspath(source))
            image_result_paths.append(os.path.join(result_dir,filename))
        elif ext in VID_FORMATS:
            video_paths.append(os.path.abspath(source))
    else:                                       # 批量视频
        for filename in os.listdir(source):
            #print(filename)
            _,ext = os.path.splitext(filename)
            #print(ext)
            if ext in IMG_FORMATS:
                image_paths.append(os.path.join(source,filename))
                image_result_paths.append(os.path.join(result_dir, filename))
            elif ext in VID_FORMATS:
                video_paths.append(os.path.join(source,filename))
    video_paths = np.array(video_paths)
    image_paths = np.array(image_paths)
    image_result_paths = np.array(image_result_paths)

    # 检测图像
    if len(image_paths) > 0:
        class_names = detection_model.get_class_names()
        colors = detection_model.get_colors()
        logger.info('共有{0}张图片需要检测'.format(len(image_paths)))
        for i, (image_path, result_path) in tqdm(enumerate(zip(image_paths, image_result_paths))):
            logger.info("开始检测图片[{0}/{1}]".format(i+1, len(image_paths)))
            image = cv2.imread(image_path)
            pred = detection_model.detect(image)
            draw_image = draw_detection_results(image, pred, class_names, colors)
            cv2.imwrite(result_path, draw_image)

    # 检测视频
    if len(video_paths) > 0:
        logger.info("共有{0}个视频需要检测".format(len(video_paths)))
        for i,video_path in tqdm(enumerate(video_paths)):
            logger.info("开始检测视频[{0}/{1}]".format(i+1,len(video_paths)))
            detection_model.detect_video(video_path, result_dir,interval)

def run_main():
    """
    这是主函数
    """
    # 初始化检测模型
    cfg = load_yaml(opt.cfg)
    model_type = cfg["DetectionModel"]["model_type"].lower()
    logger = logger_config(cfg['log_path'], model_type)
    if model_type == 'yolov5':
        from model import YOLOv5
        detection_model = YOLOv5(logger=logger, cfg=cfg)
    else:
        from model import YOLOv5
        detection_model = YOLOv5(logger=logger, cfg=cfg)

    # 初始化相关路径路径
    time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    result_dir = os.path.join(opt.result_dir,time)
    source = os.path.abspath(opt.source)
    interval = opt.interval

    # 检测
    detect(logger,detection_model,source,result_dir,interval)

if __name__ == '__main__':
    run_main()
