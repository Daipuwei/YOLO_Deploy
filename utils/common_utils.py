# -*- coding: utf-8 -*-
# @Time    : 2023/3/28 下午2:35
# @Author  : DaiPuWei
# @Email   : daipuwei@qq.com
# @File    : common_utils.py
# @Software: PyCharm

"""
    这是定义公共工具的脚本
"""

import os
import yaml
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool
from multiprocessing import cpu_count

def load_yaml(yaml_path):
    """
    这是加载ymal文件的函数
    Args:
        yaml_path: yaml文件路径
    Returns:
    """
    with open(os.path.abspath(yaml_path), 'r') as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    return cfg

def video2imageset(logger,video_paths,save_dir,bin=1):
    """
    这是将视频集进行切割并转换成图像集的函数
    Args:
        logger: 日志类实例
        video_paths: 视频文件路径数组
        save_dir: 图片保存文件夹
        bin: 间隔时长，默认为1s
    Returns:
    """
    # 多线切割视频并生成图像集
    #print(video_paths)
    #print(voc_dataset_dirs)
    size = len(video_paths)
    if size >= cpu_count():
        batch_size = size // cpu_count()
    elif size >= cpu_count() // 2:
        batch_size = size // (cpu_count() // 2)
    elif size >= cpu_count() // 4:
        batch_size = size // (cpu_count() // 4)
    else:
        batch_size = 1
    #print(size,batch_size)
    logger.info("视频抽帧开始")
    pool = Pool(processes=cpu_count())
    for start in np.arange(0,size,batch_size):
        end = int(np.min([start+batch_size,size]))
        batch_video_paths = video_paths[start:end]
        pool.apply_async(batch_videos2imagesets,callback=print_error,
                         args=(batch_video_paths,save_dir,bin))
    pool.close()
    pool.join()
    logger.info("视频抽帧结束")
    # batch_videos2voc_datasets(video_paths,voc_dataset_dirs,bin)

def print_error(value):
    """
    定义错误回调函数
    Args:
        value:
    Returns:
    """
    print("error: ", value)