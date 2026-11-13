# -*- coding: utf-8 -*-
# @Time    : 2026/9/30 11:06
# @Author  : DaiPuWei
# @Email   : daipuwei@qq.com
# @File    : yolo.py
# @Software: PyCharm

"""
    这是定义YOLO格式数据集生成器的脚本，按目录布局分为两种类型：

    模块内容：
        1. parse_yolo_label(label_path, class_names, image_path)：解析YOLO标签txt
           （每行class_id cx cy w h，归一化0-1；空行跳过），转换为与VOC/COCO
           对齐的像素级[类别名,x1,y1,x2,y2]表示；
        2. resolve_class_names_path(yolo_dataset_dir, class_names_path)：
           解析YOLO数据集类别名文件路径（自动探测或显式指定）；
        3. validate_choices(choice)：校验并规范化子集类型参数；
        4. resolve_image_path(yolo_dataset_dir, image_dir, file_name)：
           解析子集清单行对应的图像路径（yolo_v1专用）；
        5. YOLOv1Dataset：yolo_v1格式数据集类（父目录下images/labels
           两个文件夹与对应的子集txt清单文件），支持train/val/test/trainval
           四种子集；
        6. YOLOv2Dataset：yolo_v2格式数据集类（父目录下多个子集文件夹，
           每个子集文件夹内含images/labels两个文件夹），支持train/val/test/
           trainval四种子集；
        7. yolo_v1/yolo_v2(...)：数据集工厂函数，注册进DATASET_REGISTRY。
        目录约定：
        - yolo_v1：图像目录{yolo_dataset_dir}/images、标签目录
          {yolo_dataset_dir}/labels（与图像同basename的txt，每行
          class_id cx cy w h归一化0-1）、子集清单{yolo_dataset_dir}/{mode}.txt
          （每行一个图像文件名或相对数据集根目录的图像路径）；
        - yolo_v2：图像目录{yolo_dataset_dir}/{mode}/images、标签目录
          {yolo_dataset_dir}/{mode}/labels（mode为子集类型）；
        类别名文件classes.txt或obj.names（每行一个类名，位于数据集根目录），
        也可经class_names_path显式指定。
        图像无标签txt时视为无目标样本（gt为空列表）。
"""

import os
import cv2
import numpy as np

from tqdm import tqdm

from .base_dataset import Dataset
from .build import DATASET_REGISTRY
from ...detection_model.detection_utils import get_classes

# YOLO子集类型候选值
YOLO_MODES = ("train", "val", "test","trainval")
# 图像文件后缀（YOLO图像目录可能混存多种格式；含大小写变体）。
# 修复#8：曾仅认jpg/jpeg/png六种变体，bmp/tif/webp等图像被静默丢弃
# （清单行无对应图像/目录遍历跳过，均无告警）
IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp",
              ".JPG", ".JPEG", ".PNG", ".BMP", ".TIF", ".TIFF", ".WEBP")
# 类别名文件候选名（按顺序探测）
CLASS_NAMES_FILES = ("classes.txt", "obj.names","class_names.txt")


def resolve_class_names_path(yolo_dataset_dir,class_names_path=None):
    """
    这是解析YOLO数据集类别名文件路径的函数
    Args:
        yolo_dataset_dir: YOLO数据集地址
        class_names_path: 类别名文件路径，默认为None自动探测
    Returns:
        类别名文件路径
    """
    if class_names_path is not None:
        if not os.path.exists(class_names_path):
            raise FileNotFoundError("YOLO类别名文件不存在：{0}".format(class_names_path))
        return class_names_path
    for name in CLASS_NAMES_FILES:
        candidate = os.path.join(yolo_dataset_dir, name)
        if os.path.exists(candidate):
            return candidate
    raise FileNotFoundError("未找到YOLO类别名文件（{0}目录下无{1}），"
                            "请通过class_names_path显式指定"
                            .format(yolo_dataset_dir, CLASS_NAMES_FILES))


def parse_yolo_label(label_path,class_names,image_path):
    """
    这是解析YOLO标签txt文件的函数（每行class_id cx cy w h归一化0-1；
    空白行跳过——手编/拼接标签文件常见空行，与子集清单解析跳空行的
    容错口径一致；非空但字段不足的行仍显式报错）
    Args:
        label_path: 标签txt文件路径
        class_names: 类别名称数组
        image_path: 对应图像路径（读取尺寸用于归一化坐标转像素坐标）
    Returns:
        目标列表，每个目标为[类别名,x1,y1,x2,y2]（像素坐标，与VOC/COCO对齐）
    """
    gts = []
    # 读取图像尺寸（归一化坐标转像素坐标）
    image = cv2.imread(image_path)
    if image is None:
        raise IOError("图像读取失败：{0}".format(image_path))
    image_h, image_w,_ = np.shape(image)
    with open(label_path, 'r') as f:
        for line in f.readlines():
            parts = line.strip().split()
            if len(parts) == 0:
                # 空白行跳过（修复：曾对空行也走字段数校验报"行格式异常"，
                # 使整个数据集加载中止——空行是手编/拼接标签的常见形态）
                continue
            if len(parts) < 5:
                raise ValueError("YOLO标签行格式异常（期望'class_id cx cy w h'）：{0}：{1}"
                                 .format(label_path, line.strip()))
            cls_id = int(parts[0])
            cx, cy, w, h = (float(x) for x in parts[1:5])
            if cls_id < 0 or cls_id >= len(class_names):
                raise ValueError("YOLO标签类别id越界（id={0}，共{1}类）：{2}"
                                 .format(cls_id, len(class_names), label_path))
            # 归一化xywh（中心点）→ 像素xyxy（x1y1x2y2），与VOC/COCO对齐
            x1 = cx * image_w - w * image_w / 2.0
            y1 = cy * image_h - h * image_h / 2.0
            x2 = cx * image_w + w * image_w / 2.0
            y2 = cy * image_h + h * image_h / 2.0
            gts.append([class_names[cls_id], x1, y1, x2, y2])
    return gts


def validate_choices(choice):
    """
    这是校验并规范化YOLO子集类型参数的函数
    Args:
        choice: 子集类型或子集类型数组
    Returns:
        子集类型数组（按给定顺序）
    """
    choices = [choice] if isinstance(choice, str) else list(choice)
    if len(choices) == 0:
        raise ValueError("choice不能为空")
    for choice in choices:
        if choice not in YOLO_MODES:
            raise ValueError("未知的YOLO子集类型'{0}'，候选值有{1}"
                             .format(choice, YOLO_MODES))
    return choices


def resolve_image_path(yolo_dataset_dir,image_dir,file_name):
    """
    这是解析子集清单行对应图像路径的函数（yolo_v1专用，兼容三种写法：
    相对数据集根目录的路径（含images/前缀）、图像文件名、不带后缀的文件id）
    Args:
        yolo_dataset_dir: YOLO数据集地址
        image_dir: 图像目录路径（yolo_v1为images目录）
        file_name: 子集清单文件中的一行（已去空白）
    Returns:
        图像文件路径
    """
    if os.sep in file_name or "/" in file_name:
        candidate = os.path.join(yolo_dataset_dir, file_name)
        if os.path.exists(candidate):
            return candidate
        raise FileNotFoundError("子集清单中的图像路径不存在：{0}".format(candidate))
    # 图像文件名：先按原样在图像目录下查找，再按文件id探测后缀
    candidate = os.path.join(image_dir, file_name)
    if os.path.exists(candidate):
        return candidate
    file_id = os.path.splitext(file_name)[0]
    for ext in IMAGE_EXTS:
        candidate = os.path.join(image_dir, file_id + ext)
        if os.path.exists(candidate):
            return candidate
    raise FileNotFoundError("子集清单中的图像文件不存在（图像目录{0}）：{1}（支持{2}后缀）"
                            .format(image_dir, file_name, IMAGE_EXTS))


class YOLOv1Dataset(Dataset):

    def __init__(self,yolo_dataset_dir,batchsize=1,choice='train',class_names_path=None):
        """
        这是yolo_v1格式数据集生成器的初始化函数
        Args:
            yolo_dataset_dir: YOLO数据集地址（父目录下images/labels
                两个文件夹与对应的子集txt清单文件）
            batchsize: 小批量数据集规模，默认为1
            choice: 子集类型或子集类型数组，默认为'train'，
                  候选值有['train','val','test','trainval']；
                  数组形式按给定顺序合并多个子集
            class_names_path: 类别名文件路径，默认为None自动探测
                              （classes.txt或obj.names）；空行校验收口在
                              get_classes（末尾空行剔除，首部/中部空行报错）
        """
        # 类别名读取与空行校验统一收口在get_classes
        # （末尾空行剔除、首部/中部空行报错定位行号、空文件报错）
        self.class_names = get_classes(resolve_class_names_path(yolo_dataset_dir,class_names_path))
        meta_data = self.process(yolo_dataset_dir,choice)
        super(YOLOv1Dataset, self).__init__(meta_data,batchsize)

    def process(self,yolo_dataset_dir,choice='train'):
        """
        这是处理yolo_v1格式数据集的函数
        Args:
            yolo_dataset_dir: YOLO数据集地址
            choice: 子集类型或子集类型数组，默认为'train'，
                  候选值有['train','val','test','trainval']；
                  数组形式按给定顺序合并多个子集
        Returns:
            数据集元数据数组（每项含image_path与gt键，按子集清单文件行序，
            多子集按choice数组顺序拼接）
        """
        choices = validate_choices(choice)
        # 初始化相关路径
        image_dir = os.path.join(yolo_dataset_dir, "images")
        label_dir = os.path.join(yolo_dataset_dir, "labels")
        if not os.path.exists(image_dir):
            raise FileNotFoundError("yolo_v1图像目录不存在：{0}".format(image_dir))
        if not os.path.exists(label_dir):
            raise FileNotFoundError("yolo_v1标签目录不存在：{0}".format(label_dir))
        # 逐个处理子集（多子集按choice数组顺序拼接）
        meta_data = []
        for choice in choices:
            txt_path = os.path.join(yolo_dataset_dir, choice + ".txt")
            if not os.path.exists(txt_path):
                raise FileNotFoundError("yolo_v1子集清单文件不存在：{0}".format(txt_path))
            with open(txt_path, 'r') as f:
                for line in tqdm(f.readlines(), desc="解析YOLOv1标签[{0}]".format(choice)):
                    file_name = line.strip()
                    if not file_name:
                        continue
                    image_path = resolve_image_path(yolo_dataset_dir, image_dir, file_name)
                    file_id = os.path.splitext(os.path.basename(image_path))[0]
                    label_path = os.path.join(label_dir, file_id + ".txt")
                    if not os.path.exists(label_path):
                        # 无标签txt的图像视为无目标样本（YOLO数据集的合法形态）
                        gts = []
                    else:
                        gts = parse_yolo_label(label_path, self.class_names, image_path)
                    meta_data.append({
                        "image_path": image_path,
                        "gt": gts,
                    })
        meta_data = np.array(meta_data)
        return meta_data


class YOLOv2Dataset(Dataset):

    def __init__(self,yolo_dataset_dir,batchsize=1,choice='train',class_names_path=None):
        """
        这是yolo_v2格式数据集生成器的初始化函数
        Args:
            yolo_dataset_dir: YOLO数据集地址（父目录下多个子集文件夹，
                每个子集文件夹内含images/labels两个文件夹）
            batchsize: 小批量数据集规模，默认为1
            choice: 子集类型或子集类型数组，默认为'train'，
                  候选值有['train','val','test','trainval']；
                  数组形式按给定顺序合并多个子集
            class_names_path: 类别名文件路径，默认为None自动探测
                              （classes.txt或obj.names）；空行校验收口在
                              get_classes（末尾空行剔除，首部/中部空行报错）
        """
        # 类别名读取与空行校验统一收口在get_classes
        # （末尾空行剔除、首部/中部空行报错定位行号、空文件报错）
        self.class_names = get_classes(resolve_class_names_path(yolo_dataset_dir,class_names_path))
        meta_data = self.process(yolo_dataset_dir,choice)
        super(YOLOv2Dataset, self).__init__(meta_data,batchsize)

    def process(self,yolo_dataset_dir,choice='train'):
        """
        这是处理yolo_v2格式数据集的函数
        Args:
            yolo_dataset_dir: YOLO数据集地址
            choice: 子集类型或子集类型数组，默认为'train'，
                  候选值有['train','val','test','trainval']；
                  数组形式按给定顺序合并多个子集
        Returns:
            数据集元数据数组（每项含image_path与gt键，按图像文件名字典序，
            多子集按choice数组顺序拼接）
        """
        choices = validate_choices(choice)
        # 逐个处理子集（多子集按choice数组顺序拼接）
        meta_data = []
        for choice in choices:
            # 初始化相关路径（子集文件夹内含images/labels两个目录）
            image_dir = os.path.join(yolo_dataset_dir, choice, "images")
            label_dir = os.path.join(yolo_dataset_dir, choice, "labels")
            if not os.path.exists(image_dir):
                raise FileNotFoundError("yolo_v2图像目录不存在：{0}".format(image_dir))
            if not os.path.exists(label_dir):
                raise FileNotFoundError("yolo_v2标签目录不存在：{0}".format(label_dir))
            # 遍历图像目录（字典序保证批次顺序确定），对应标签txt解析
            for file_name in tqdm(sorted(os.listdir(image_dir)),
                                  desc="解析YOLOv2标签[{0}]".format(choice)):
                if not file_name.endswith(IMAGE_EXTS):
                    continue
                image_path = os.path.join(image_dir, file_name)
                file_id = os.path.splitext(file_name)[0]
                label_path = os.path.join(label_dir, file_id + ".txt")
                if not os.path.exists(label_path):
                    # 无标签txt的图像视为无目标样本（YOLO数据集的合法形态）
                    gts = []
                else:
                    gts = parse_yolo_label(label_path, self.class_names, image_path)
                meta_data.append({
                    "image_path": image_path,
                    "gt": gts,
                })
        meta_data = np.array(meta_data)
        return meta_data


@DATASET_REGISTRY.register()
def yolo_v1(dataset_dir,batchsize=1,choice='train',class_names_path=None):
    """
    这是yolo_v1格式数据集类的注册函数
    Args:
        dataset_dir: 数据集路径（父目录下images/labels两个文件夹
            与对应的子集txt清单文件）
        batchsize: 小批量数据规模，默认为1
        choice: 子集类型，默认为'train'，候选值有['train','val','test','trainval']
        class_names_path: 类别名文件路径，默认为None自动探测
    Returns:
        yolo_v1数据集加载器
    """
    dataloader = YOLOv1Dataset(dataset_dir,batchsize,choice,class_names_path)
    return dataloader


@DATASET_REGISTRY.register()
def yolo_v2(dataset_dir,batchsize=1,choice='train',class_names_path=None):
    """
    这是yolo_v2格式数据集类的注册函数
    Args:
        dataset_dir: 数据集路径（父目录下多个子集文件夹，
            每个子集文件夹内含images/labels两个文件夹）
        batchsize: 小批量数据规模，默认为1
        choice: 子集类型，默认为'train'，候选值有['train','val','test','trainval']
        class_names_path: 类别名文件路径，默认为None自动探测
    Returns:
        yolo_v2数据集加载器
    """
    dataloader = YOLOv2Dataset(dataset_dir,batchsize,choice,class_names_path)
    return dataloader
