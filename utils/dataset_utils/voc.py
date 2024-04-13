# -*- coding: utf-8 -*-
# @Time    : 2024/4/13 下午10:21
# @Author  : DaiPuWei
# @Email   : daipuwei@qq.com
# @File    : voc.py
# @Software: PyCharm

"""
    这是定义VOC数据集生成器的脚本
"""

import os
import numpy as np
import xml.etree.ElementTree as ET

from .base_dataset import Dataset
from .build import DATASET_REGISTRY

def is_contain_object(xml_path):
    """
    这是判断XML文件中是否包含目标标签的函数
    :param xml_path: XML文件路径
    :return:
    """
    # 获取XML文件的根结点
    root = ET.parse(xml_path).getroot()
    return len(root.findall('object')) > 0

def parse_xml(xml_path):
    """
    这是解析VOC数据集XML标签文件，获取每个目标分类与定位的函数
    :param xml_path: XML标签文件路径
    :return:
    """
    # 获取XML文件的根结点
    root = ET.parse(xml_path).getroot()
    # 遍历所有目标
    objects = []
    for obj in root.findall('object'):
        obj_name = obj.find('name').text
        bndbox = obj.find('bndbox')
        xmin = int(bndbox.find('xmin').text)
        ymin = int(bndbox.find('ymin').text)
        xmax = int(bndbox.find('xmax').text)
        ymax = int(bndbox.find('ymax').text)
        objects.append([obj_name, xmin,ymin,xmax-xmin,ymax-ymin])
    return objects

class VOCDataset(Dataset):

    def __init__(self,voc_dataset_dir,batchsize=1,mode='train'):
        """
        这是VOC数据集生成器的初始化函数
        Args:
            voc_dataset_dir: VOC数据集地址
            batchsize: 小批量数据集规模，默认为1
            mode: 数据集类型，默认为‘train’,侯选值有['train','val','trainval','test']
        """
        meta_data = self.process(voc_dataset_dir,mode)
        super(VOCDataset, self).__init__(meta_data,batchsize)
    def process(self,voc_dataset_dir,mode='train'):
        """
        这是处理VOC数据集的函数
        Args:
            voc_dataset_dir:voc数据集地址
            mode: 子集类型，默认为'train'
        Returns:
        """
        # 初始化相关路径
        voc_annotation_dir = os.path.join(voc_dataset_dir,'Annotations')
        voc_image_dir = os.path.join(voc_dataset_dir,"JPEGImages")
        if not os.path.exists(voc_image_dir):
            voc_image_dir = os.path.join(voc_dataset_dir,"images")
        voc_main_dir = os.path.join(voc_dataset_dir,"ImageSets","Main")
        train_txt_path = os.path.join(voc_main_dir,'train.txt')
        val_txt_path = os.path.join(voc_main_dir,'val.txt')
        test_txt_path = os.path.join(voc_main_dir, 'test.txt')
        trainval_txt_path = os.path.join(voc_main_dir, 'trainval.txt')
        if mode == "train":
            txt_path = train_txt_path
        elif mode == 'val':
            txt_path = val_txt_path
        elif mode == 'test':
            txt_path = test_txt_path
        else:
            txt_path = trainval_txt_path

        # 初始化图像和标签文件路径
        meta_data = []
        with open(txt_path,'r') as f:
            for line in f.readlines():
                image_path = os.path.join(voc_image_dir,line.strip()+".jpg")
                xml_path = os.path.join(voc_annotation_dir,line.strip()+".xml")
                gts = parse_xml(xml_path)
                meta_data.append({
                    "image_path": image_path,
                    "gt": gts,
                })
        meta_data = np.array(meta_data)
        return meta_data

@DATASET_REGISTRY.register()
def voc(dataset_dir,batchsize=1,mode='train'):
    """
    这是VOC数据集类的注册函数
    Args:
        dataset_dir: 数据集路径
        batchsize: 小批量数据规模，默认为1
        mode: 子集类型，默认为‘train’
    Returns:
    """
    dataloader = VOCDataset(dataset_dir,batchsize,mode)
    return dataloader
