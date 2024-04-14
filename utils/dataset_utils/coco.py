# -*- coding: utf-8 -*-
# @Time    : 2024/4/13 下午10:21
# @Author  : DaiPuWei
# @Email   : daipuwei@qq.com
# @File    : coco.py
# @Software: PyCharm

"""
    这是定义COCO数据集加载器脚本
"""

import os
import cv2
import json
import numpy as np
from .base_dataset import Dataset
from .build import DATASET_REGISTRY

class COCODataset(Dataset):

    def __init__(self,coco_dataset_dir,batchsize=1,mode='train'):
        """
        这是COCO数据集生成器的初始化函数
        Args:
            coco_dataset_dir: COCO数据集地址
            batchsize: 小批量数据集规模，默认为1
            mode: 数据集类型，默认为‘train’,侯选值有['train','val','trainval','test']
        """
        meta_data = self.process(coco_dataset_dir,mode)
        super(COCODataset,self).__init__(meta_data,batchsize)

    def process(self, coco_dataset_dir,mode="train"):
        """
        这是解析coco数据集的函数
        Args:
            coco_dataset_dir: coco数据集地址
            mode: 子集类型，默认为'train'
        Returns:
        """
        # 初始化相关路径
        coco_image_dir = os.path.join(coco_dataset_dir, mode)
        coco_json_path = os.path.join(coco_dataset_dir, 'annotations', '{}.json'.format(mode))

        # 初始化图像和标签文件路径
        image_gt_dict = {}
        image_id_dict = {}
        id_category_dict = {}
        with open(coco_json_path, "r", encoding='utf-8') as f:
            json_data = json.load(f)
            image_infos = json_data['images']
            gts = json_data['annotations']
            category_id_dict_list = json_data['categories']
            for _dict in category_id_dict_list:
                id_category_dict[_dict['id']] = _dict["name"]
            for image_info in image_infos:
                image_id_dict[image_info['id']] = os.path.join(coco_image_dir, image_info['file_name'])
            for gt in gts:
                image_id = gt['image_id']
                x1, y1, w, h = gt['bbox']
                cls_id = gt['category_id']
                image_path = image_id_dict[image_id]
                if image_path not in image_gt_dict.keys():
                    image_gt_dict[image_path] = [[id_category_dict[cls_id], x1, y1, w, h]]
                else:
                    image_gt_dict[image_path].append([id_category_dict[cls_id], x1, y1, w, h])

        # 初始化meta数据
        meta_data = []
        for image_path,gts in image_gt_dict.items():
            meta_data.append({
                "image_path": image_path,
                "gt": gts,
            })
        meta_data = np.array(meta_data)
        return meta_data

@DATASET_REGISTRY.register()
def coco(dataset_dir,batchsize=1,mode='val'):
    """
    这是COCO数据集类的注册函数
    Args:
        dataset_dir: 数据集路径
        batchsize: 小批量数据规模，默认为1
        mode: 子集类型，默认为‘train’
    Returns:
    """
    dataloader = COCODataset(dataset_dir,batchsize,mode)
    return dataloader
