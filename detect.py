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
import sys
import cv2
import numpy as np

from tqdm import tqdm
from threading import Thread
from datetime import datetime
from model import build_model

from utils import ArgsParser
from utils import init_config
from utils import init_logger
from utils import draw_detection_results

IMG_FORMATS = ['.bmp', '.dng', '.jpeg', '.jpg', '.mpo', '.png', '.tif', '.tiff', '.webp']  # include image suffixes
VID_FORMATS = ['.asf', '.avi', '.gif', '.m4v', '.mkv', '.mov', '.mp4', '.mpeg', '.mpg', '.ts', '.wmv','.dav']  # include video suffixes

parser = ArgsParser()
parser.add_argument('--cfg', type=str, default='./config/detection.yaml', help='config yaml file path')
parser.add_argument('--source', type=str, default='./video', help='image path or image directory or video path or video directory')
parser.add_argument('--result_dir', type=str, default="./result", help='video detection result save directory')
parser.add_argument('--interval', type=float, default=-1, help='video interval')
parser.add_argument('--num_threads', type=int, default=1, help='number of detection threads')
parser.add_argument('--gpu_id', type=int, default=0, help='gpu id')
parser.add_argument('--print_detection_result', action='store_true', help='export time')
opt = parser.parse_args()

def detect(logger,detection_models,source,result_dir,interval=1,print_detection_result=False):
    """
    这是利用检测模型检测图像或者视频的函数
    Args:
        logger: 日志类
        detection_models: 检测模型列表
        source: 图像文件或者图像文件夹或者视频文件路径或者视频文件夹
        result_dir: 结果保存路径
        interval: 视频抽帧间隔,默认为1s
        print_detection_result: 是否打印检测结果，默认为False
    Returns:
    """
    # 初始化视频路径
    result_dir = os.path.abspath(result_dir)
    image_result_dir = os.path.join(result_dir,"image")
    video_result_dir = os.path.join(result_dir,"video")
    if not os.path.exists(image_result_dir):
        os.makedirs(image_result_dir)
    if not os.path.exists(video_result_dir):
        os.makedirs(video_result_dir)

    video_paths = []
    video_result_paths = []
    image_paths = []
    image_result_paths = []
    if os.path.isfile(source):               # 单个视频
        _,filename = os.path.split(source)
        fname,ext = os.path.splitext(filename)
        if ext in IMG_FORMATS:
            image_paths.append(os.path.abspath(source))
            image_result_paths.append(os.path.join(image_result_dir,fname+".jpg"))
        elif ext in VID_FORMATS:
            video_paths.append(os.path.abspath(source))
            video_result_paths.append(os.path.join(video_result_dir, fname + ".mp4"))
    else:                                       # 批量视频
        for filename in os.listdir(source):
            #print(filename)
            fname,ext = os.path.splitext(filename)
            #print(ext)
            if ext in IMG_FORMATS:
                image_paths.append(os.path.join(source,filename))
                image_result_paths.append(os.path.join(image_result_dir, fname+".jpg"))
            elif ext in VID_FORMATS:
                video_paths.append(os.path.join(source,filename))
                video_result_paths.append(os.path.join(video_result_dir,fname+".mp4"))
    video_paths = np.array(video_paths)
    video_result_paths = np.array(video_result_paths)
    image_paths = np.array(image_paths)
    image_result_paths = np.array(image_result_paths)

    # 检测图像
    if len(image_paths) > 0:
        logger.info('开始检测图像，共有{0}张图片需要检测'.format(len(image_paths)))
        detect_imageset(detection_models,image_paths,image_result_paths,print_detection_result)
        logger.info('结束检测图像')

    # 检测视频
    if len(video_paths) > 0:
        logger.info("开始检测视频，共有{0}个视频需要检测".format(len(video_paths)))
        detect_videoset(detection_models,video_paths,video_result_paths,interval,print_detection_result)
        logger.info('结束检测图像')

def detect_imageset(detection_models,image_paths,image_result_paths,print_detection_result=False):
    """
    这是检测图像集的函数
    Args:
        detection_models: 检测模型列表
        image_paths: 图像路径数组
        image_result_paths: 检测结果图像路径数组
        print_detection_result: 是否打印检测结果，默认为False
    Returns:
    """
    # 多线程检测图像并保存检测结果
    size = len(image_paths)
    num_models = len(detection_models)
    if size // num_models != 0:
        num_threads = num_models
    elif size // (num_models // 2) != 0:
        num_threads = num_models // 2
    elif size // (num_models // 4) != 0:
        num_threads = num_models // 4
    else:
        num_threads = 1
    for i in np.arange(num_models-num_threads):
        del detection_models[0]
    batch_size = size // num_threads
    start = 0
    threads = []
    for i in np.arange(num_threads):
        if i != num_threads-1:
            end = start + batch_size
        else:
            end = size
        batch_image_paths = image_paths[start:end]
        batch_image_result_paths = image_result_paths[start:end]
        detection_model = detection_models[i]
        start = end
        t = Thread(target=detect_batch_images,
                   args=(detection_model,batch_image_paths,
                         batch_image_result_paths,print_detection_result))
        t.start()
        threads.append(t)
    for t in threads:
        t.join()

def detect_batch_images(detection_model,batch_image_paths,batch_image_result_paths,print_detection_result=False):
    """
    这是检测批量图像的函数
    Args:
        detection_model: 检测模型
        batch_image_paths: 批量图像路径数组
        batch_image_result_paths: 批量检测结果图像路径数组
        print_detection_result: 是否打印检测结果，默认为False
    Returns:
    """
    # 遍历所有图片进行检测
    size = len(batch_image_paths)
    for i in tqdm(np.arange(size)):
        detect_single_image(detection_model,
                            batch_image_paths[i],batch_image_result_paths[i],print_detection_result)
def detect_single_image(detection_model,image_path,image_result_path,print_detection_result=False):
    """
    这是检测单张图像的函数
    Args:
        detection_model: 检测模型
        image_path: 图像路径
        image_result_path: 检测结果图像路径
        print_detection_result: 是否打印检测结果，默认为False
    Returns:
    """
    # 检测视频并绘制检测结果
    image = cv2.imread(image_path)
    pred = detection_model.detect(image, print_detection_result=print_detection_result)

    # 绘制检测结果
    colors = detection_model.get_colors()
    draw_image = draw_detection_results(image, pred, colors)
    cv2.imwrite(image_result_path, draw_image)

def detect_videoset(detection_models,video_paths,video_result_paths,interval=-1,print_detection_result=False):
    """
    这是检测视频集的函数
    Args:
        detection_models: 检测模型列表
        video_paths: 视频路径数组
        video_result_paths: 检测结果视频路径数组
        interval: 视频抽帧频率,默认为-1,逐帧检测
        print_detection_result: 是否打印检测结果，默认为False
    Returns:
    """
    # 多线程检测图像并保存检测结果
    size = len(video_paths)
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
    detection_models = detection_models[:num_threads]
    start = 0
    threads = []
    for i in np.arange(num_threads):
        if i != num_threads-1:
            end = start + batch_size
        else:
            end = size
        batch_video_paths = video_paths[start:end]
        batch_video_result_paths = video_result_paths[start:end]
        detection_model = detection_models[i]
        start = end
        t = Thread(target=detect_batch_videos,
                   args=(detection_model,batch_video_paths,
                         batch_video_result_paths,interval,print_detection_result))
        t.start()
        threads.append(t)
    for t in threads:
        t.join()

def detect_batch_videos(detection_model, batch_video_paths,
                        batch_video_result_paths,interval=-1, print_detection_result=False):
    """
    这是检测批量图像的函数
    Args:
        detection_model: 检测模型
        batch_video_paths: 批量视频路径数组
        batch_video_result_paths: 批量检测结果视频路径数组
        interval: 视频抽帧频率,默认为-1,逐帧检测
        print_detection_result: 是否打印检测结果，默认为False
    Returns:
    """
    # 遍历所有图片进行检测
    size = len(batch_video_paths)
    for i in tqdm(np.arange(size)):
        detect_single_video(detection_model,batch_video_paths[i],batch_video_result_paths[i], interval,print_detection_result)

def detect_single_video(detection_model, video_path, video_result_path,interval=-1,print_detection_result=False):
    """
    这是检测单张图像的函数
    Args:
        detection_model: 检测模型
        video_path: 视频路径
        video_result_path: 检测结果视频路径
        interval: 视频抽帧频率,默认为-1,逐帧检测
        print_detection_result: 是否打印检测结果，默认为False
    Returns:
    """
    # 检测视频，并保存检测结果
    detection_model.detect_video(video_path,video_result_path,interval,print_detection_result)

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
        detection_model = build_model(logger,cfg["DetectionModel"],gpu_id=opt.gpu_id)
        detection_models.append(detection_model)

    # 初始化相关路径路径
    time = datetime.now().strftime('%Y%m%d%H%M%S')
    model_path = cfg["DetectionModel"]["engine_model_path"]
    _,model_name = os.path.split(model_path)
    result_dir = os.path.join(opt.result_dir, model_name,time)
    source = os.path.abspath(opt.source)
    interval = opt.interval

    # 检测
    detect(logger,detection_models,source,result_dir,interval,opt.print_detection_result)

if __name__ == '__main__':
    run_main()
