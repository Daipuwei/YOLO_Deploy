# -*- coding: utf-8 -*-
# @Time    : 2023/4/8 下午9:32
# @Author  : DaiPuWei
# @Email   : daipuwei@qq.com
# @File    : dataloader_utils.py
# @Software: PyCharm

"""
    这是定义数据集生成器的脚本
"""

import os
import json
import cv2
import numpy as np
import xml.etree.ElementTree as ET

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

class VOCDataset(object):

    def __init__(self,voc_dataset_dir,batchsize=1,mode='train'):
        """
        这是VOC数据集生成器的初始化函数
        Args:
            voc_dataset_dir: VOC数据集地址
            batchsize: 小批量数据集规模，默认为1
            mode: 数据集类型，默认为‘train’,侯选值有['train','val','trainval','test']
        """
        # 初始化相关路径
        self.voc_dataset_dir = os.path.abspath(voc_dataset_dir)
        self.batchsize = batchsize
        self.voc_annotation_dir = os.path.join(self.voc_dataset_dir,'Annotations')
        image_dir = os.path.join(self.voc_dataset_dir,"JPEGImages")
        if os.path.exists(image_dir):
            self.voc_image_dir = image_dir
        else:
            self.voc_image_dir = os.path.join(self.voc_dataset_dir,"images")
        self.voc_main_dir = os.path.join(self.voc_dataset_dir,"ImageSets","Main")
        self.train_txt_path = os.path.join(self.voc_main_dir,'train.txt')
        self.val_txt_path = os.path.join(self.voc_main_dir,'val.txt')
        self.test_txt_path = os.path.join(self.voc_main_dir, 'test.txt')
        self.trainval_txt_path = os.path.join(self.voc_main_dir, 'trainval.txt')
        if mode == "train":
            self.txt_path = self.train_txt_path
        elif mode == 'val':
            self.txt_path = self.val_txt_path
        elif mode == 'test':
            self.txt_path = self.test_txt_path
        else:
            self.txt_path = self.trainval_txt_path

        self.start = 0
        self.end = 0

        # 初始化图像和标签文件路径
        self.image_paths = []
        self.xml_paths = []
        with open(self.txt_path,'r') as f:
            for line in f.readlines():
                image_path = os.path.join(self.voc_image_dir,line.strip()+".jpg")
                xml_path = os.path.join(self.voc_annotation_dir,line.strip()+".xml")
                self.image_paths.append(image_path)
                self.xml_paths.append(xml_path)
        self.image_paths = np.array(self.image_paths)
        self.xml_paths = np.array(self.xml_paths)
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
        batch_xml_paths = self.xml_paths[start:end]
        batch_images = [cv2.imread(image_path) for image_path in batch_image_paths]
        batch_gts = [parse_xml(xml_path) for xml_path in batch_xml_paths]
        return batch_images,batch_gts,batch_image_paths

    def __next__(self):
        self.end = min(self.start+self.batchsize,self.image_num)
        if self.end == self.image_num:
            self.start = 0
            self.end = 0
            raise StopIteration
        else:
            batch_images, batch_gts,batch_image_paths = self.get_batch_data(self.start,self.end)
            self.start = self.end
            return batch_images, batch_gts,batch_image_paths

    def __iter__(self):
        return self

    def get_image_num(self):
        return self.image_num

class COCODataset(object):

    def __init__(self,coco_dataset_dir,batchsize=1,mode='train'):
        """
        这是COCO数据集生成器的初始化函数
        Args:
            coco_dataset_dir: COCO数据集地址
            batchsize: 小批量数据集规模，默认为1
            mode: 数据集类型，默认为‘train’,侯选值有['train','val','trainval','test']
        """
        # 初始化相关路径
        self.coco_dataset_dir = os.path.abspath(coco_dataset_dir)
        self.batchsize = batchsize
        self.train_image_dir = os.path.join(self.coco_dataset_dir,'train')
        self.val_image_dir = os.path.join(self.coco_dataset_dir,'val')
        self.test_image_dir = os.path.join(self.coco_dataset_dir,'test')
        self.test_image_dir = os.path.join(self.coco_dataset_dir, 'test')
        self.val_json_path = os.path.join(self.coco_dataset_dir, 'annotations', 'val.json')
        self.test_json_path = os.path.join(self.coco_dataset_dir, 'annotations', 'test.json')
        if mode == "train":
            self.image_dirs = [self.train_image_dir]
            self.json_paths = [self.train_json_path]
        elif mode == 'val':
            self.image_dirs = [self.val_image_dir]
            self.json_paths = [self.val_json_path]
        elif mode == 'test':
            self.image_dirs = [self.test_image_dir]
            self.json_paths = [self.test_json_path]
        else:
            self.image_dirs = [self.train_image_dir,self.val_image_dir]
            self.json_paths = [self.train_json_path,self.val_json_path]

        self.start = 0
        self.end = 0

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
        batch_images = [cv2.imread(image_path) for image_path in batch_image_paths]
        batch_gts = [self.image_gt_dict[image_path] for image_path in batch_image_paths]
        return batch_images,batch_gts,batch_image_paths

    def __next__(self):
        self.end = min(self.start+self.batchsize,self.image_num)
        if self.end == self.image_num:
            self.start = 0
            self.end = 0
            raise StopIteration
        else:
            batch_images, batch_gts,batch_image_paths = self.get_batch_data(self.start,self.end)
            self.start = self.end
            return batch_images, batch_gts,batch_image_paths

    def __iter__(self):
        return self

    def get_image_num(self):
        return self.image_num
