# -*- coding: utf-8 -*-
# @Time    : 2023/3/28 下午3:47
# @Author  : DaiPuWei
# @Email   : daipuwei@qq.com
# @File    : video2voc_dataset.py
# @Software: PyCharm

"""
    这是将视频转换为VOC数据集的脚本
"""

import os
import sys
import cv2
import numpy as np

from tqdm import tqdm
from threading import Thread
from datetime import datetime
from multiprocessing import Pool
from multiprocessing import cpu_count
from pascal_voc_writer import Writer

from utils import ArgsParser
from utils import init_config
from utils import print_error
from utils import logger_config

VID_FORMATS = ['.asf', '.avi', '.gif', '.m4v', '.mkv', '.mov', '.mp4', '.mpeg', '.mpg', '.ts', '.wmv']  # include video suffixes

parser = ArgsParser()
parser.add_argument('--cfg', type=str, default='./config/detection.yaml', help='config yaml file path')
parser.add_argument('--video', type=str, default='./video', help='video path or video directory')
parser.add_argument('--result_dir', type=str, default="./result/voc_dataset", help='voc dataset save directory')
parser.add_argument('--interval', type=float, default=-1, help='video interval')
parser.add_argument('--num_threads', type=int, default=1, help='number of detection threads')
parser.add_argument('--print_detection_result', action='store_true', help='export time')
opt = parser.parse_args()

def video2voc_dataset(logger,detection_model,video,result_dir,interval=1,num_threads=-1,print_detection_result=False):
    """
    这是利用检测模型对视频(集)进行预标注并生成VOC数据集的函数
    Args:
        logger: 日志类实例
        detection_model: 检测模型实例
        video: 视频文件路径或者视频文件夹
        result_dir: 结果保存路径
        interval: 视频抽帧间隔，默认为1s
        num_threads: 检测线程数,默认为1
        print_detection_result: 是否打印检测结果，默认为False
    Returns:
    """
    # 初始化视频路径
    result_dir = os.path.abspath(result_dir)
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    voc_dataset_dirs = []
    #print(source)
    logger.info("开始初始化视频文件")
    if os.path.isfile(video):               # 单个视频
        video_paths = [os.path.abspath(video)]
    else:
        video_paths = []
        for filename in os.listdir(video):
            fname,ext = os.path.splitext(filename)
            if ext in VID_FORMATS:
                video_paths.append(os.path.join(video,filename))
    for video_path in video_paths:
        _,video_name = os.path.split(video_path)
        fname, ext = os.path.splitext(video_name)
        voc_dataset_dir = os.path.join(result_dir, fname)
        voc_image_dir = os.path.join(voc_dataset_dir, "JPEGImages")
        voc_xml_dir = os.path.join(voc_dataset_dir, "Annotations")
        if not os.path.exists(voc_image_dir):
            os.makedirs(voc_image_dir)
        if not os.path.exists(voc_xml_dir):
            os.makedirs(voc_xml_dir)
        voc_dataset_dirs.append(voc_dataset_dir)
    video_paths = np.array(video_paths)
    voc_dataset_dirs = np.array(voc_dataset_dirs)
    logger.info("结束初始化视频文件")
    logger.info("共有{}个视频需要转换成VOC数据集".format(len(voc_dataset_dirs)))

    # ffmpeg对视频进行抽帧
    logger.info("视频抽帧开始")
    video_decode(video_paths,voc_dataset_dirs,interval)
    logger.info("视频抽帧结束")

    # 初始化图像路径和XML路径
    voc_image_paths = []
    voc_xml_paths = []
    for voc_dataset_dir in voc_dataset_dirs:
        voc_image_dir = os.path.join(voc_dataset_dir,"JPEGImages")
        voc_annotation_dir = os.path.join(voc_dataset_dir,"Annotations")
        if not os.path.exists(voc_image_dir):
            os.makedirs(voc_image_dir)
        if not os.path.exists(voc_annotation_dir):
            os.makedirs(voc_annotation_dir)
        for image_name in os.listdir(voc_image_dir):
            fname,ext = os.path.splitext(image_name)
            voc_image_paths.append(os.path.join(voc_image_dir,image_name))
            voc_xml_paths.append(os.path.join(voc_annotation_dir,fname+".xml"))
    voc_image_paths = np.array(voc_image_paths)
    voc_xml_paths = np.array(voc_xml_paths)
    logger.info("解析得到图片{0}张".format(len(voc_image_paths)))

    # 检测图像并生成VOC数据集标签
    logger.info("图像检测与预标注开始")
    prelabel_imageset_save_voc_dataset(detection_model,voc_image_paths,voc_xml_paths,num_threads,print_detection_result)
    logger.info("图像检测与预标注结束")

def video_decode(video_paths, voc_dataset_dirs,interval=-1):
    """
    这是对视频进行解码生成VOC数据集的函数
    Args:
        video_paths: 视频路径数组
        voc_dataset_dirs: VOC数据集路径数组
        interval: 视频抽帧频率,默认为-1，
    Returns:
    """
    # 多线切割视频并生成图像集
    size = len(video_paths)
    if size // cpu_count() != 0:
        num_pools = cpu_count()
    elif size // (cpu_count() // 2) != 0:
        num_pools = cpu_count() // 2
    elif size // (cpu_count() // 4) !=0:
        num_pools = cpu_count() // 4
    else:
        num_pools = 1
    batch_size = size // num_pools
    pool = Pool(processes=num_pools)
    for start in np.arange(0,size,batch_size):
        end = int(np.min([start+batch_size,size]))
        batch_video_paths = video_paths[start:end]
        batch_voc_dataset_dirs = voc_dataset_dirs[start:end]
        pool.apply_async(batch_videos2voc_dataset,callback=print_error,
                         args=(batch_video_paths,batch_voc_dataset_dirs,interval))
    pool.close()
    pool.join()

def batch_videos2voc_dataset(batch_video_paths,batch_voc_dataset_paths,interval=-1):
    """
    这是将批量视频进行切割并转换VOC数据集的函数
    Args:
        batch_video_paths: 批量视频文件名路径数组
        batch_voc_dataset_paths: 批量VOC数据集路径数组
        interval: 间隔时长 ，默认为1s
    Return:
    """
    for i in tqdm(np.arange(len(batch_video_paths))):
        single_video2voc_dataset(batch_video_paths[i],batch_voc_dataset_paths[i],interval)

def single_video2voc_dataset(video_path,voc_dataset_path,interval=-1):
    """
    这是将单个视频进行切割并转换为VOC数据集的函数
    Args:
        video_path: 视频文件路径
        voc_dataset_path: VOC数据集路径
        interval: 间隔时长，默认为-1
    Returns:
    """
    # 初始化视频名称
    _, video_name = os.path.split(video_path)
    fname, _ = os.path.splitext(video_name)

    # 读取视频fps
    vid_cap = cv2.VideoCapture(video_path)
    fps = int(round(vid_cap.get(cv2.CAP_PROP_FPS)))
    if int(interval) == -1:
        bin = 1
    else:
        bin = int(round(fps*interval))

    # 利用FFmpeg进行视频抽帧
    image_format = os.path.join(voc_dataset_path,"JPEGImages","{0}_frame%08d.jpg".format(fname))
    os.system("ffmpeg -i {0} -qscale:v 1 -r {1} {2}".format(video_path, bin, image_format))

def prelabel_imageset_save_voc_dataset(detection_model,voc_image_paths,voc_xml_paths,num_threads=1,print_detection_result=False):
    """
    这是检测图像集并保存VOC标签集的函数
    Args:
        detection_model: 检测模型实例
        voc_image_paths: VOC图像文件路径数组
        voc_xml_paths: VOC标签文件路径数组
        num_threads: 检测线程数,默认为1
        print_detection_result: 是否打印检测结果,默认为False
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
        batch_voc_image_paths = voc_image_paths[start:end]
        batch_voc_xml_paths = voc_xml_paths[start:end]
        _detection_model = detection_model[i]
        start = end
        t = Thread(target=detect_batch_images,
                   args=(_detection_model,batch_voc_image_paths,batch_voc_xml_paths,print_detection_result))
        t.start()
        threads.append(t)
    for t in threads:
        t.join()

def detect_batch_images(detection_model,batch_voc_image_paths,batch_voc_xml_paths,print_detection_result=False):
    """
    这是利用检测模型检测批量图像并生成VOC标签的函数
    Args:
        detection_model: 检测模型
        batch_voc_image_paths: 批量VOC图像文件路径数组
        batch_voc_xml_paths: 批量VOC标签文件路径数组
        print_detection_result: 是否打印检测结果，默认为False
    Returns:
    """
    for i in tqdm(np.arange(len(batch_voc_image_paths))):
        detect_single_image(detection_model,batch_voc_image_paths[i],batch_voc_xml_paths[i],print_detection_result)

def detect_single_image(detection_model,voc_image_path,voc_xml_path,print_detection_result=False):
    """
    这是利用检测模型检测单张图像并保存VOC数据标签的函数
    Args:
        detection_model: 检测模型实例
        voc_image_path: VOC图像文件路径
        voc_xml_path: VOC标签文件路径
        print_detection_result: 是否打印检测结果，默认为False
    Returns:
    """
    # 初始化VOC标签写入类
    image = cv2.imread(voc_image_path)
    h, w, _ = np.shape(image)
    writer = Writer(voc_image_path,w,h)

    # 检测图像并将检测结果写入XML
    preds = detection_model.detect(image,export_time=False,print_detection_result=print_detection_result)
    if len(preds) > 0:
        for pred in preds:
            x1, y1, x2, y2 = pred['bbox']
            cls_name = pred['cls_name']
            x1 = int(round(x1))
            y1 = int(round(y1))
            x2 = int(round(x2))
            y2 = int(round(y2))
            writer.addObject(cls_name, x1, y1, x2, y2)
        writer.save(voc_xml_path)

def run_main():
    """
    这是主函数
    """
    # 初始化参数
    cfg = init_config(opt)
    # 初始化检测模型
    model_type = cfg["DetectionModel"]["model_type"].lower()
    model_path = cfg["DetectionModel"]["engine_model_path"]
    _,model_name = os.path.split(model_path)
    logger = logger_config(cfg['log_path'], model_type)
    detection_models = []
    for _ in np.arange(opt.num_threads):
        if model_type == 'yolov5':
            from model import YOLOv5
            detection_model = YOLOv5(logger=logger, cfg=cfg["DetectionModel"])
        elif model_type == 'yolov8':
            from model import YOLOv8
            detection_model = YOLOv8(logger=logger, cfg=cfg["DetectionModel"])
        elif model_type == 'yolos':
            from model import YOLOS
            detection_model = YOLOS(logger=logger, cfg=cfg["DetectionModel"])
        else:
            sys.exit(0)
        detection_models.append(detection_model)

    # 初始化图像及其结果保存文件夹路径
    time = datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
    result_dir = os.path.join(opt.result_dir,model_name,time)
    video = os.path.abspath(opt.video)

    # 检测图像
    video2voc_dataset(logger, detection_models, video,result_dir,opt.interval,opt.num_threads,opt.print_detection_result)

if __name__ == '__main__':
    run_main()
