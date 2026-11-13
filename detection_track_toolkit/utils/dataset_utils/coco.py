
# -*- coding: utf-8 -*-
# @Time    : 2026/9/10 11:06
# @Author  : DaiPuWei
# @Email   : daipuwei@qq.com
# @File    : coco.py
# @Software: PyCharm

"""
    这是定义COCO数据集加载器的脚本。

    模块内容：
        1. COCODataset：COCO数据集类（解析COCO目录并构建元数据，
           批量加载复用Dataset基类），支持train/val/test/trainval四种子集；
        2. coco(...)：数据集工厂函数，注册进DATASET_REGISTRY（注册名'coco'）。
        目录约定（本仓库定制，非官方COCO目录）：图像目录{coco_dataset_dir}/{mode}，
        标注文件{coco_dataset_dir}/annotations/{mode}.json。
        图像无标注时视为无目标样本（gt为空列表，与VOC/YOLO约定一致）。
"""

import os
import json
import numpy as np

from tqdm import tqdm

from .base_dataset import Dataset
from .build import DATASET_REGISTRY

# COCO子集类型候选值
COCO_MODES = ("train", "val", "test", "trainval")
# 图像文件后缀探测顺序（json中的file_name可能缺后缀或与磁盘实际后缀
# 不符；含大小写变体）。修复#8：曾仅认jpg/jpeg/png六种变体，file_name
# 缺后缀而磁盘为bmp/tif/webp的图像探测失败被静默跳过
IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp",
              ".JPG", ".JPEG", ".PNG", ".BMP", ".TIF", ".TIFF", ".WEBP")

class COCODataset(Dataset):

    def __init__(self,coco_dataset_dir,batchsize=1,choice='train'):
        """
        这是COCO数据集生成器的初始化函数
        Args:
            coco_dataset_dir: COCO数据集地址
            batchsize: 小批量数据集规模，默认为1
            choice: 子集类型或子集类型数组，默认为'train'，
                  候选值有['train','val','test','trainval']；
                  数组形式按给定顺序合并多个子集
        """
        meta_data = self.process(coco_dataset_dir,choice)
        super(COCODataset,self).__init__(meta_data,batchsize)

    def process(self, coco_dataset_dir,choice="train"):
        """
        这是解析coco数据集的函数
        Args:
            coco_dataset_dir: coco数据集地址
            choice: 子集类型或子集类型数组，默认为'train'，
                  候选值有['train','val','test','trainval']；
                  数组形式按给定顺序合并多个子集
        Returns:
            数据集元数据数组（每项含image_path与gt键，按标注json的图像顺序，
            无标注的图像gt为空列表；多子集按choice数组顺序拼接）
        """
        choices = [choice] if isinstance(choice, str) else list(choice)
        if len(choices) == 0:
            raise ValueError("choice不能为空")
        for choice in choices:
            if choice not in COCO_MODES:
                raise ValueError("未知的COCO子集类型'{0}'，候选值有{1}"
                                 .format(choice, COCO_MODES))

        # 逐个处理子集（多子集按choice数组顺序拼接）
        meta_data = []
        for choice in choices:
            # 初始化相关路径
            coco_image_dir = os.path.join(coco_dataset_dir, choice)
            coco_json_path = os.path.join(coco_dataset_dir, 'annotations', '{}.json'.format(choice))
            if not os.path.exists(coco_image_dir):
                raise FileNotFoundError("COCO图像目录不存在：{0}".format(coco_image_dir))
            if not os.path.exists(coco_json_path):
                raise FileNotFoundError("COCO标注文件不存在：{0}".format(coco_json_path))

            # 初始化图像和标签文件路径
            image_gt_dict = {}
            image_id_dict = {}
            id_category_dict = {}
            with open(coco_json_path, "r", encoding='utf-8') as f:
                json_data = json.load(f)
                for key in ("images", "annotations", "categories"):
                    if key not in json_data:
                        raise ValueError("COCO标注json缺少'{0}'字段：{1}"
                                         .format(key, coco_json_path))
                # 初始化目标名称与目标id字典
                category_id_dict_list = json_data['categories']
                for _dict in category_id_dict_list:
                    id_category_dict[_dict['id']] = _dict["name"]

                # 遍历图像信息字典构造图像路径与图像id字典
                image_infos = json_data['images']
                for image_info in tqdm(image_infos, desc="探测COCO图像[{0}]".format(choice)):
                    # 探测图像文件：先按json中的file_name原样查找，
                    # 找不到时按常见后缀探测（file_name可能缺后缀或与磁盘后缀不符）
                    image_path = os.path.join(coco_image_dir, image_info['file_name'])
                    if not os.path.exists(image_path):
                        image_path = None
                        base_name = os.path.splitext(image_info['file_name'])[0]
                        for ext in IMAGE_EXTS:
                            candidate = os.path.join(coco_image_dir, base_name + ext)
                            if os.path.exists(candidate):
                                image_path = candidate
                                break
                    if image_path is None:
                        raise FileNotFoundError("COCO图像文件不存在（图像目录{0}）：{1}（支持{2}后缀）"
                                                .format(coco_image_dir, image_info['file_name'],
                                                        IMAGE_EXTS))
                    image_id_dict[image_info['id']] = image_path

                # 遍历所有标签，构造图像路径与标签字典
                gts = json_data['annotations']
                for gt in tqdm(gts, desc="解析COCO标注[{0}]".format(choice)):
                    image_id = gt['image_id']
                    bbox = gt['bbox']
                    if len(bbox) != 4:
                        raise ValueError("COCO标注bbox格式异常（期望4个元素，实际{0}个）：image_id={1}"
                                         .format(len(bbox), image_id))
                    x1, y1, w, h = bbox
                    cls_id = gt['category_id']
                    if image_id not in image_id_dict:
                        raise ValueError("COCO标注引用了不存在的图像：image_id={0}".format(image_id))
                    if cls_id not in id_category_dict:
                        raise ValueError("COCO标注引用了不存在的类别：category_id={0}".format(cls_id))
                    image_path = image_id_dict[image_id]
                    # bbox为xywh，统一转换为x1y1x2y2（与VOC/YOLO对齐）
                    gt_info = [id_category_dict[cls_id], x1, y1, x1 + w, y1 + h]
                    if image_path not in image_gt_dict.keys():
                        image_gt_dict[image_path] = [gt_info]
                    else:
                        image_gt_dict[image_path].append(gt_info)

            # 初始化meta数据：按标注json的图像顺序遍历（不再遍历无序字典，
            # 保证批次内图像顺序确定、可复现）；无标注的图像视为无目标样本
            # （gt为空列表，与VOC/YOLO约定一致）
            for image_info in image_infos:
                image_path = image_id_dict[image_info['id']]
                meta_data.append({
                    "image_path": image_path,
                    "gt": image_gt_dict.get(image_path, []),
                })
        meta_data = np.array(meta_data)
        return meta_data

@DATASET_REGISTRY.register()
def coco(dataset_dir,batchsize=1,choice='train'):
    """
    这是COCO数据集类的注册函数
    Args:
        dataset_dir: 数据集路径
        batchsize: 小批量数据规模，默认为1
        choice: 子集类型，默认为'train'，候选值有['train','val','test','trainval']
    Returns:
    """
    dataloader = COCODataset(dataset_dir,batchsize,choice)
    return dataloader
