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
        super(COCODataset,self).__init__(coco_dataset_dir,batchsize,mode)
        # 初始化相关路径
        self.train_image_dir = os.path.join(self.dataset_dir,'train')
        self.val_image_dir = os.path.join(self.dataset_dir,'val')
        self.test_image_dir = os.path.join(self.dataset_dir,'test')
        self.test_image_dir = os.path.join(self.dataset_dir, 'test')
        self.val_json_path = os.path.join(self.dataset_dir, 'annotations', 'val.json')
        self.test_json_path = os.path.join(self.dataset_dir, 'annotations', 'test.json')
        if self.mode == "train":
            self.image_dirs = [self.train_image_dir]
            self.json_paths = [self.train_json_path]
        elif self.mode == 'val':
            self.image_dirs = [self.val_image_dir]
            self.json_paths = [self.val_json_path]
        elif self.mode == 'test':
            self.image_dirs = [self.test_image_dir]
            self.json_paths = [self.test_json_path]
        else:
            self.image_dirs = [self.train_image_dir,self.val_image_dir]
            self.json_paths = [self.train_json_path,self.val_json_path]

        # 初始化图像和标签文件路径
        self.image_gt_dict = {}
        self.image_id_dict = {}
        self.id_category_dict = {}
        for image_dir,json_path in zip(self.image_dirs,self.json_paths):
            with open(json_path, "r",encoding='utf-8') as f:
                json_data = json.load(f)
                image_infos = json_data['images']
                gts = json_data['annotations']
                category_id_dict_list = json_data['categories']
                for _dict in category_id_dict_list:
                    self.id_category_dict[_dict['id']] = _dict["name"]
                for image_info in image_infos:
                    self.image_id_dict[image_info['id']] = os.path.join(image_dir,image_info['file_name'])
                for gt in gts:
                    image_id= gt['image_id']
                    x1,y1,w,h = gt['bbox']
                    cls_id = gt['category_id']
                    image_path = self.image_id_dict[image_id]
                    if image_path not in self.image_gt_dict.keys():
                        self.image_gt_dict[image_path] = [[self.id_category_dict[cls_id], x1, y1, w, h]]
                    else:
                        self.image_gt_dict[image_path].append([self.id_category_dict[cls_id], x1, y1, w, h])
        self.image_paths = np.array(list(self.image_gt_dict.keys()))
        self.image_num = len(self.image_paths)

    def __len__(self):
        size = self.image_num // self.batchsize
        if self.image_num % self.batchsize != 0:
            size += 1
        return size

    def get_batch_data(self,start,end):
        """
        这是加载一个batchsize数据的函数
        Args:
            start: 开始索引
            end: 结束索引
        Returns:
        """
        batch_image_paths = self.image_paths[start:end]
        batch_meta_data = []
        batch_images = []
        for image_path in batch_image_paths:
            image = cv2.imread(image_path)
            gts = self.image_gt_dict[image_path]
            batch_meta_data.append({
                "gt": gts,
                "image_path": image_path
            })
            batch_images.append(image)
        batch_images = np.array(batch_images)
        batch_meta_data = np.array(batch_meta_data)
        return batch_images,batch_meta_data

    def __next__(self):
        self.end = min(self.start+self.batchsize,self.image_num)
        if self.end == self.image_num:
            self.start = 0
            self.end = 0
            raise StopIteration
        else:
            batch_images,batch_meta_data = self.get_batch_data(self.start,self.end)
            self.start = self.end
            return batch_images,batch_meta_data

    def __iter__(self):
        return self

    def get_image_num(self):
        return self.image_num

@DATASET_REGISTRY.register()
def coco(dataset_dir,batchsize=1,mode='train'):
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
