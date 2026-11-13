#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2026/9/11 01:59
# @Author  : DaiPuWei
# @Email   : daipuwei@qq.com
# @File    : detect.py
# @Software: PyCharm

"""
    这是检测图像和视频的脚本。

    模块内容：
        1. detect(logger, detection_models, source, result_dir, interval,
           print_detection_result)：检测总入口，支持图像/视频的单文件或目录
           （按后缀自动分流），结果统一归入result_dir下的image/与video/子目录
           （图像结果统一保存为jpg，视频结果统一保存为mp4）；
        2. detect_imageset/detect_videoset：多线程调度函数（每线程一个模型
           实例，线程数直接按传入num_threads分配、小于1强制为1，
           不修改传入的模型列表）；
        3. detect_batch_images/detect_batch_videos：批量遍历函数（tqdm进度条）；
        4. detect_single_image/detect_single_video：单任务检测与保存函数
           （读图失败显式报错）；
        5. scan_media_files(source, image_result_dir, video_result_dir)：
           扫描媒体文件函数（单文件或递归目录，扩展名大小写不敏感）；
        6. run_main：主函数（命令行参数在函数内解析，模块可安全import）。
"""

import os
import sys
import cv2
import numpy as np

from tqdm import tqdm
from threading import Thread
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
from detection_track_toolkit.utils import ArgsParser
from detection_track_toolkit.utils import init_config
from detection_track_toolkit.utils import init_logger

IMG_FORMATS = ['.bmp', '.dng', '.jpeg', '.jpg', '.mpo', '.png', '.tif', '.tiff', '.webp']  # include image suffixes
VID_FORMATS = ['.asf', '.avi', '.gif', '.m4v', '.mkv', '.mov', '.mp4', '.mpeg', '.mpg', '.ts', '.wmv','.dav']  # include video suffixes

def scan_media_files(source,image_result_dir,video_result_dir):
    """
    这是扫描待检测媒体文件并生成结果保存路径的函数。
    源路径与结果目录统一规范为绝对路径；目录递归遍历时结果路径保留源目录
    相对层级（在结果目录下重建子目录），彻底避免下划线扁平化撞名
    （曾将子目录相对路径replace(os.sep,"_")拼入文件名，a/b/x.jpg与
    a_b/x.jpg压平成同名结果文件互相覆盖）。扩展名比较大小写不敏感
    （修复：曾按小写表原样比较，.JPG/.PNG/.MP4等大写后缀被静默跳过）。
    Args:
        source: 图像/视频文件路径或其所在目录
        image_result_dir: 图像结果保存目录
        video_result_dir: 视频结果保存目录
    Returns:
        (image_paths, image_result_paths, video_paths, video_result_paths)：
        图像/视频的源路径（绝对）与结果路径（绝对，保留源目录相对层级）数组
    """
    source = os.path.abspath(source)
    image_result_dir = os.path.abspath(image_result_dir)
    video_result_dir = os.path.abspath(video_result_dir)
    video_paths = []
    video_result_paths = []
    image_paths = []
    image_result_paths = []

    def classify(path,fname,rel_dir):
        # 扩展名统一小写比较（大小写不敏感），结果统一保存为jpg/mp4；
        # 主名用splitext取（不能用小写ext做rfind——大写后缀文件rfind
        # 返回-1会把主名截错，如a.JPG→a.JP）；结果路径保留rel_dir层级
        # 并预先创建父目录（cv2.imwrite/VideoWriter不自动建目录）
        stem, ext = os.path.splitext(fname)
        ext = ext.lower()
        if ext in IMG_FORMATS:
            subdir = image_result_dir if rel_dir == "" \
                else os.path.join(image_result_dir, rel_dir)
            os.makedirs(subdir, exist_ok=True)
            image_paths.append(os.path.abspath(path))
            image_result_paths.append(os.path.join(subdir, stem + ".jpg"))
        elif ext in VID_FORMATS:
            subdir = video_result_dir if rel_dir == "" \
                else os.path.join(video_result_dir, rel_dir)
            os.makedirs(subdir, exist_ok=True)
            video_paths.append(os.path.abspath(path))
            video_result_paths.append(os.path.join(subdir, stem + ".mp4"))

    if os.path.isfile(source):               # 单个文件
        _,filename = os.path.split(source)
        classify(source,filename,"")
    else:                                       # 批量文件（递归遍历子目录）
        # 递归遍历：源目录下多层子目录中的图像/视频都参与检测；
        # 结果路径保留源目录相对层级（rel_dir），不同子目录同名文件
        # 落到不同结果子目录，不再互相覆盖
        for dirpath, _, filenames in sorted(os.walk(source)):
            rel_dir = os.path.relpath(dirpath, source)
            rel_dir = "" if rel_dir == "." else rel_dir
            for filename in sorted(filenames):
                classify(os.path.join(dirpath, filename), filename, rel_dir)
    video_paths = np.array(video_paths)
    video_result_paths = np.array(video_result_paths)
    image_paths = np.array(image_paths)
    image_result_paths = np.array(image_result_paths)
    return image_paths,image_result_paths,video_paths,video_result_paths


def detect(logger,detection_models,source,result_dir,interval=1,print_detection_result=False,num_threads=1,export_time=False):
    """
    这是利用检测模型检测图像或者视频的函数
    Args:
        logger: 日志类
        detection_models: 检测模型列表
        source: 图像文件或者图像文件夹或者视频文件路径或者视频文件夹
        result_dir: 结果保存路径
        interval: 视频抽帧间隔,默认为1s
        print_detection_result: 是否打印检测结果，默认为False
        num_threads: 检测线程数，默认为1（小于1强制为1）
        export_time: 是否导出检测时间，默认为False
    Returns:
    """
    # 初始化结果路径
    result_dir = os.path.abspath(result_dir)
    image_result_dir = os.path.join(result_dir,"image")
    video_result_dir = os.path.join(result_dir,"video")

    image_paths,image_result_paths,video_paths,video_result_paths = \
        scan_media_files(source,image_result_dir,video_result_dir)

    # 无任何媒体文件：打告警提示（修复：曾两个if都不进、静默no-op，
    # 误传空目录/错误路径时用户以为在跑检测实际什么都没做）
    if len(image_paths) == 0 and len(video_paths) == 0:
        logger.warning("在{0}下未找到任何图像/视频文件，无检测任务执行".format(source))
        return

    # 检测图像（结果目录按需创建）
    if len(image_paths) > 0:
        if not os.path.exists(image_result_dir):
            os.makedirs(image_result_dir)
        logger.info('开始检测图像，共有{0}张图片需要检测'.format(len(image_paths)))
        detect_imageset(detection_models,image_paths,image_result_paths,
                        print_detection_result,num_threads,export_time)
        logger.info('结束检测图像')

    # 检测视频
    if len(video_paths) > 0:
        if not os.path.exists(video_result_dir):
            os.makedirs(video_result_dir)
        logger.info("开始检测视频，共有{0}个视频需要检测".format(len(video_paths)))
        detect_videoset(detection_models,video_paths,video_result_paths,
                        interval,print_detection_result,num_threads,export_time)
        logger.info('结束检测视频')

def detect_imageset(detection_models,image_paths,image_result_paths,print_detection_result=False,num_threads=1,export_time=False):
    """
    这是检测图像集的函数
    Args:
        detection_models: 检测模型列表
        image_paths: 图像路径数组
        image_result_paths: 检测结果图像路径数组
        print_detection_result: 是否打印检测结果，默认为False
        num_threads: 检测线程数，默认为1（小于1强制为1；调用方需保证
                     模型数量不少于num_threads，run_main按num_threads构建模型）
        export_time: 是否导出检测时间，默认为False
    Returns:
    """
    # 多线程检测图像并保存检测结果：线程数直接按传入值分配（小于1强制为1），
    # 每线程使用一个独立的模型实例（不修改传入的模型列表）
    size = len(image_paths)
    if size == 0:
        return
    num_threads = max(1, num_threads)
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
                         batch_image_result_paths,print_detection_result,
                         export_time))
        t.start()
        threads.append(t)
    for t in threads:
        t.join()

def detect_batch_images(detection_model,batch_image_paths,batch_image_result_paths,print_detection_result=False,export_time=False):
    """
    这是检测批量图像的函数
    Args:
        detection_model: 检测模型
        batch_image_paths: 批量图像路径数组
        batch_image_result_paths: 批量检测结果图像路径数组
        print_detection_result: 是否打印检测结果，默认为False
        export_time: 是否导出检测时间，默认为False
    Returns:
    """
    # 遍历所有图片进行检测
    size = len(batch_image_paths)
    for i in tqdm(np.arange(size)):
        detect_single_image(detection_model,
                            batch_image_paths[i],batch_image_result_paths[i],
                            print_detection_result,export_time)

def detect_single_image(detection_model,image_path,image_result_path,print_detection_result=False,export_time=False):
    """
    这是检测单张图像的函数
    Args:
        detection_model: 检测模型
        image_path: 图像路径
        image_result_path: 检测结果图像路径
        print_detection_result: 是否打印检测结果，默认为False
        export_time: 是否导出检测时间，默认为False
    Returns:
    """
    # 检测图像并绘制检测结果
    image = cv2.imread(image_path)
    if image is None:
        raise IOError("图像读取失败：{0}".format(image_path))
    pred = detection_model.detect(image, export_time=export_time,
                                  print_detection_result=print_detection_result)
    # export_time为True时detect返回(检测结果,时间信息字典)，拆包取检测结果
    if export_time:
        pred = pred[0]

    # 绘制检测结果
    colors = detection_model.get_colors()
    draw_image = draw_detection_results(image, pred, colors)
    cv2.imwrite(image_result_path, draw_image)

def detect_videoset(detection_models,video_paths,video_result_paths,interval=-1,print_detection_result=False,num_threads=1,export_time=False):
    """
    这是检测视频集的函数
    Args:
        detection_models: 检测模型列表
        video_paths: 视频路径数组
        video_result_paths: 检测结果视频路径数组
        interval: 视频抽帧频率,默认为-1,逐帧检测
        print_detection_result: 是否打印检测结果，默认为False
        num_threads: 检测线程数，默认为1（小于1强制为1；调用方需保证
                     模型数量不少于num_threads，run_main按num_threads构建模型）
        export_time: 是否导出检测时间，默认为False
    Returns:
    """
    # 多线程检测视频并保存检测结果：线程数直接按传入值分配（小于1强制为1），
    # 每线程使用一个独立的模型实例（不修改传入的模型列表）
    size = len(video_paths)
    if size == 0:
        return
    num_threads = max(1, num_threads)
    batch_size = size // num_threads
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
        t = Thread(target=detect_batch_videoes,
                   args=(detection_model,batch_video_paths,
                         batch_video_result_paths,interval,print_detection_result,
                         export_time))
        t.start()
        threads.append(t)
    for t in threads:
        t.join()

def detect_batch_videoes(detection_model, batch_video_paths,
                        batch_video_result_paths,interval=-1, print_detection_result=False,
                        export_time=False):
    """
    这是检测批量视频的函数
    Args:
        detection_model: 检测模型
        batch_video_paths: 批量视频路径数组
        batch_video_result_paths: 批量检测结果视频路径数组
        interval: 视频抽帧频率,默认为-1,逐帧检测
        print_detection_result: 是否打印检测结果，默认为False
        export_time: 是否导出检测时间，默认为False
    Returns:
    """
    # 遍历所有视频进行检测
    size = len(batch_video_paths)
    for i in tqdm(np.arange(size)):
        detect_single_video(detection_model,batch_video_paths[i],batch_video_result_paths[i],
                           interval,print_detection_result,export_time)

def detect_single_video(detection_model, video_path, video_result_path,interval=-1,print_detection_result=False,export_time=False):
    """
    这是检测单段视频的函数
    Args:
        detection_model: 检测模型
        video_path: 视频路径
        video_result_path: 检测结果视频路径
        interval: 视频抽帧频率,默认为-1,逐帧检测
        print_detection_result: 是否打印检测结果，默认为False
        export_time: 是否导出检测时间，默认为False
    Returns:
    """
    # 检测视频，并保存检测结果
    detection_model.detect_video(video_path,video_result_path,interval,
                                 print_detection_result,export_time)

def run_main():
    """
    这是主函数
    """
    # 解析命令行参数，并初始化参数
    parser = ArgsParser()
    parser.add_argument('--cfg', type=str, default=os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), 'config', 'detection.yaml'), help='config yaml file path')
    parser.add_argument('--source', type=str, default='./video',
                        help='image path or image directory or video path or video directory')
    parser.add_argument('--result_dir', type=str, default="./result/detect", help='video detection result save directory')
    parser.add_argument('--interval', type=float, default=-1, help='video interval')
    parser.add_argument('--num_threads', type=int, default=1, help='number of detection threads')
    parser.add_argument('--gpu_id', type=int, default=0, help='gpu id (multiple threads share this gpu)')
    parser.add_argument('--print_detection_result', action='store_true', help='print detection result')
    parser.add_argument('--export_time', action='store_true', help='export detection time')
    opt = parser.parse_args()
    cfg = init_config(opt)

    # 初始化日志
    logger = init_logger(cfg["Log"])

    # 参数修正：线程数非法（<=0）时直接置为1，
    if opt.num_threads <= 0:
        logger.warning("num_threads非法（{0}），已置为1".format(opt.num_threads))
        opt.num_threads = 1

    # 初始化检测模型（多线程共用同一gpu_id）
    detection_models = []
    for _ in np.arange(opt.num_threads):
        detection_model = build_model(logger,cfg["DetectionModel"],gpu_id=opt.gpu_id)
        detection_models.append(detection_model)

    # 初始化相关路径
    time_str = datetime.now().strftime('%Y%m%d%H%M%S')
    model_path = cfg["DetectionModel"]["engine_model_path"]
    _,model_name = os.path.split(model_path)
    result_dir = os.path.join(opt.result_dir, model_name,time_str)
    source = os.path.abspath(opt.source)
    interval = opt.interval

    # 检测
    detect(logger,detection_models,source,result_dir,interval,
           opt.print_detection_result,opt.num_threads,opt.export_time)

    # 释放检测模型资源
    for detection_model in detection_models:
        detection_model.release()

if __name__ == '__main__':
    run_main()
