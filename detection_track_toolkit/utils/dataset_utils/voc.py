# -*- coding: utf-8 -*-
# @Time    : 2026/9/10 11:06
# @Author  : DaiPuWei
# @Email   : daipuwei@qq.com
# @File    : voc.py
# @Software: PyCharm

"""
    这是定义VOC数据集生成器的脚本。

    模块内容：
        1. parse_xml(xml_path)：解析VOC标注XML，返回目标列表
           （每项为[类别名,x1,y1,x2,y2]的xyxy内部表示，三种数据集格式统一）；
        2. VOCDataset：VOC数据集类（解析VOC目录并构建元数据，
           批量加载复用Dataset基类），支持train/val/test/trainval四种子集；
        3. voc(...)：数据集工厂函数，注册进DATASET_REGISTRY（注册名'voc'）。
        目录约定：{voc_dataset_dir}/Annotations（标签XML）、
        {voc_dataset_dir}/JPEGImages或images（图像）、
        {voc_dataset_dir}/ImageSets/Main/{mode}.txt（子集清单）。
        图像无标签XML时视为无目标样本（gt为空列表，与YOLO/COCO约定一致）。
"""

import os
import numpy as np
import xml.etree.ElementTree as ET

from tqdm import tqdm

from .base_dataset import Dataset
from .build import DATASET_REGISTRY

# VOC子集类型候选值
VOC_MODES = ("train", "val", "test", "trainval")
# 图像文件后缀探测顺序（VOC图像集可能混存多种格式；含大小写变体）。
# 修复#8：曾仅认jpg/jpeg/png六种变体，bmp/tif/webp等图像探测失败
# 被静默跳过（配合缺图[WARN]告警，后缀表外图像现已可正常命中）
IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp",
              ".JPG", ".JPEG", ".PNG", ".BMP", ".TIF", ".TIFF", ".WEBP")


def parse_xml(xml_path):
    """
    这是解析VOC数据集XML标签文件的函数
    Args:
        xml_path: XML标签文件路径
    Returns:
        目标列表，每个目标为[类别名,x1,y1,x2,y2]
    """
    # 获取XML文件的根结点
    root = ET.parse(xml_path).getroot()
    # 遍历所有目标
    objects = []
    for obj in root.findall('object'):
        name_node = obj.find('name')
        bndbox = obj.find('bndbox')
        if name_node is None or bndbox is None:
            raise ValueError("VOC标签XML结构异常（缺少name或bndbox节点）：{0}"
                             .format(xml_path))
        obj_name = name_node.text
        xmin = int(bndbox.find('xmin').text)
        ymin = int(bndbox.find('ymin').text)
        xmax = int(bndbox.find('xmax').text)
        ymax = int(bndbox.find('ymax').text)
        objects.append([obj_name,xmin,ymin,xmax,ymax])
    return objects

class VOCDataset(Dataset):

    def __init__(self,voc_dataset_dir,batchsize=1,choice='train'):
        """
        这是VOC数据集生成器的初始化函数
        Args:
            voc_dataset_dir: VOC数据集地址
            batchsize: 小批量数据集规模，默认为1
            choice: 子集类型或子集类型数组，默认为'train'，
                  候选值有['train','val','test','trainval']；
                  数组形式按给定顺序合并多个子集
        """
        meta_data = self.process(voc_dataset_dir,choice)
        super(VOCDataset, self).__init__(meta_data,batchsize)

    def process(self,voc_dataset_dir,choice='train'):
        """
        这是处理VOC数据集的函数。子集清单行兼容裸id（0001，VOC官方惯例）
        与自带后缀（0001.jpg，迁移自YOLO等格式时常见）两种写法：
        统一splitext去后缀后探测图像（IMAGE_EXTS逐个后缀）与xml
        （按去后缀id拼.xml，避免带后缀行被误判为无目标样本）；
        空行跳过；清单行无对应图像时打印[WARN]告警后跳过（不静默丢数据）
        Args:
            voc_dataset_dir: voc数据集地址
            choice: 子集类型或子集类型数组，默认为'train'，
                  候选值有['train','val','test','trainval']；
                  数组形式按给定顺序合并多个子集
        Returns:
            数据集元数据数组（每项含image_path与gt键，无标签XML的图像gt为空列表，
            多子集按choice数组顺序拼接）
        """
        choices = [choice] if isinstance(choice, str) else list(choice)
        if len(choices) == 0:
            raise ValueError("choice不能为空")
        for choice in choices:
            if choice not in VOC_MODES:
                raise ValueError("未知的VOC子集类型'{0}'，候选值有{1}"
                                 .format(choice, VOC_MODES))
        # 初始化相关路径
        voc_annotation_dir = os.path.join(voc_dataset_dir,'Annotations')
        voc_image_dir = os.path.join(voc_dataset_dir,"JPEGImages")
        if not os.path.exists(voc_image_dir):
            voc_image_dir = os.path.join(voc_dataset_dir,"images")
        voc_main_dir = os.path.join(voc_dataset_dir,"ImageSets","Main")

        # 逐个处理子集并初始化图像和标签文件路径（多子集按choice数组顺序拼接）
        meta_data = []
        for choice in choices:
            txt_path = os.path.join(voc_main_dir, choice + '.txt')
            if not os.path.exists(txt_path):
                raise FileNotFoundError("VOC子集清单文件不存在：{0}".format(txt_path))
            with open(txt_path,'r') as f:
                for line in tqdm(f.readlines(), desc="解析VOC标注[{0}]".format(choice)):
                    file_id = line.strip()
                    if not file_id:
                        # 空行跳过（与yolo.py子集清单解析口径一致）
                        continue
                    # 清单行可能自带后缀（如0001.jpg，迁移自YOLO等格式时常见）：
                    # 统一splitext去后缀后再探测，裸id行不受影响（splitext原样返回）
                    stem = os.path.splitext(file_id)[0]
                    # 探测图像后缀（VOC图像集可能混存jpg/png等格式，不再硬编码.jpg）
                    image_path = None
                    for ext in IMAGE_EXTS:
                        candidate = os.path.join(voc_image_dir, stem + ext)
                        if os.path.exists(candidate):
                            image_path = candidate
                            break
                    if image_path is None:
                        # 未找到对应图像：打印告警后跳过（不静默丢数据；
                        # 清单与图像目录增删不同步时保持可排查）
                        print("[WARN] VOC清单行'{0}'未找到对应图像（图像目录{1}，"
                              "支持{2}后缀），已跳过"
                              .format(file_id, voc_image_dir, IMAGE_EXTS))
                        continue
                    # 解析xml文件（同样按去后缀id解析，清单行带后缀时命中0001.xml
                    # 而非0001.jpg.xml，避免带后缀行被误判为无目标样本
                    xml_path = os.path.join(voc_annotation_dir, stem + ".xml")
                    if not os.path.exists(xml_path):
                        # 无标签XML的图像视为无目标样本（与YOLO/COCO约定一致）
                        gts = []
                    else:
                        gts = parse_xml(xml_path)
                    meta_data.append({
                        "image_path": image_path,
                        "gt": gts,
                    })
        meta_data = np.array(meta_data)
        return meta_data

@DATASET_REGISTRY.register()
def voc(dataset_dir,batchsize=1,choice='train'):
    """
    这是VOC数据集类的注册函数
    Args:
        dataset_dir: 数据集路径
        batchsize: 小批量数据规模，默认为1
        choice: 子集类型，默认为'train'，候选值有['train','val','test','trainval']
    Returns:
    """
    dataloader = VOCDataset(dataset_dir,batchsize,choice)
    return dataloader
