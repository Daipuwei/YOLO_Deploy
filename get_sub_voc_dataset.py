# -*- coding: utf-8 -*-
# @Time    : 2024/3/26 下午10:12
# @Author  : DaiPuWei
# @Email   : daipuwei@qq.com
# @File    : get_sub_voc_dataset.py
# @Software: PyCharm

"""
    这是获取指定VOC子集图像的脚本
"""

import os
import json
import shutil
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool,cpu_count

def get_sub_voc_dataset(voc_dataset_dir,imageset_dir,choices=['val']):
    """
    这是获取指定VOC子集图像的函数
    Args:
        voc_dataset_dir: voc数据集地址
        imageset_dir: 图像集地址
        choices: voc子集候选列表，默认为["val"]
    Returns:
    """
    # 初始化相关路径
    voc_image_dir = os.path.join(voc_dataset_dir,'JPEGImages')
    voc_main_dir = os.path.join(voc_dataset_dir,"ImageSets","Main")
    if not os.path.exists(imageset_dir):
        os.makedirs(imageset_dir)

    # 初始化图像及其json标签文件路径
    voc_image_paths = []
    imageset_image_paths = []
    for choice in choices:
        choice_txt_path = os.path.join(voc_main_dir,choice+".txt")
        with open(choice_txt_path,'r') as f:
            for line in f.readlines():
                voc_image_paths.append(os.path.join(voc_image_dir,line.strip()+".jpg"))
                imageset_image_paths.append(os.path.join(imageset_dir,line.strip()+".jpg"))
    voc_image_paths = np.array(voc_image_paths)
    imageset_image_paths = np.array(imageset_image_paths)

    # 多线程处理复制图像和json文件
    size = len(voc_image_paths)
    if size > 0:
        # batch_size = size // cpu_count()
        if size // cpu_count() != 0:
            num_threads = cpu_count()
        elif size // (cpu_count() // 2) != 0:
            num_threads = cpu_count() // 2
        elif size // (cpu_count() // 4) != 0:
            num_threads = cpu_count() // 4
        else:
            num_threads = 1
        batch_size = size // num_threads
        pool = Pool(processes=num_threads)
        for start in np.arange(0, size, batch_size):
            end = int(np.min([start + batch_size, size]))
            batch_voc_image_paths = voc_image_paths[start:end]
            batch_imageset_image_paths = imageset_image_paths[start:end]
            pool.apply_async(copy_batch_images, error_callback=print_error,
                             args=(batch_voc_image_paths,batch_imageset_image_paths))
        pool.close()
        pool.join()

def copy_batch_images(batch_voc_image_paths,batch_imageset_image_paths):
    """
    这是复制批量图像的函数
    Args:
        batch_voc_image_paths: 批量VOC数据集图像文件路径数组
        batch_imageset_image_paths: 批量图像集图像文件路径数组
    Returns:
    """
    for i in tqdm(np.arange(len(batch_voc_image_paths))):
        copy_single_image(batch_voc_image_paths[i],batch_imageset_image_paths[i])

def copy_single_image(voc_image_path, imageset_image_path):
    """
    这是复制单张图像的函数
    Args:
        voc_image_path: voc数据集图像路径
        imageset_image_path: 图像集图像路径
    Returns:
    """
    shutil.copy(voc_image_path,imageset_image_path)

def print_error(value):
    """
    定义错误回调函数
    Args:
        value:
    Returns:
    """
    print("error: ", value)


def run_main():
    """
    这是主函数
    """
    voc_dataset_dir = os.path.abspath("/home/dpw/deeplearning/coco2017_voc")
    imageset_dir = os.path.abspath("/home/dpw/deeplearning/coco_calib")
    get_sub_voc_dataset(voc_dataset_dir,imageset_dir)

if __name__ == '__main__':
    run_main()