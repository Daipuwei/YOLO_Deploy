# -*- coding: utf-8 -*-
# @Time    : 2026/9/10 08:48
# @Author  : DaiPuWei
# @Email   : daipuwei@qq.com
# @File    : detection_utils.py
# @Software: PyCharm

import cv2
import colorsys
import numpy as np

def get_classes(classes_path):
    """
    这是获取目标分类名称的函数（行号即类别id，空行处理统一在此收口）：
    末尾空行剔除（文件末尾多余回车产生的空串，无害）；
    首部/中部空行显式报错并定位行号——行号即类别id，保留则类名为空、
    剔除则后续类别id整体前移，两者都导致id→类名映射错乱（静默错标）；
    空文件（含全空白行）报错
    Args:
        classes_path: 目标分类名称txt文件路径
    Returns:
        类别名称数组（保证非空且不含空字符串）
    """
    classes_names = []
    with open(classes_path, 'r') as f:
        for line in f.readlines():
            classes_names.append(line.strip())
    # 剔除末尾空行（文件末尾多余回车产生的空字符串，无害）
    while classes_names and not classes_names[-1]:
        classes_names.pop()
    # 首部/中部空行：显式报错并定位行号（fail-fast，避免静默错标）
    for idx, name in enumerate(classes_names):
        if not name:
            raise ValueError("类别名文件第{0}行为空行：行号即类别id，空行会导致"
                             "类别id错位（id前移/越界/静默错标），请删除该空行：{1}"
                             .format(idx + 1, classes_path))
    if len(classes_names) == 0:
        raise ValueError("类别名文件为空：{0}".format(classes_path))
    return classes_names


def get_anchors(anchors_path):
    """
    这是获取模型anchor宽高数据的函数（yolov5多输出原始模型融合解码用，
    读取方式与get_classes一致：txt每行一个数值）
    Args:
        anchors_path: anchors文件路径（每行一个数值，按anchor宽高成对排列，
            如10/13/16/30/33/23/...）
    Returns:
        anchor宽高数组，shape为(anchor数,2)，每行为[宽,高]（按输出层数
        拆分由调用方根据engine输出节点个数完成）
    """
    vals = []
    with open(anchors_path, 'r') as f:
        for line in f.readlines():
            line = line.strip()
            if line:
                try:
                    vals.append(float(line))
                except ValueError:
                    raise ValueError("anchors文件每行必须为单个数值，实际为{0}"
                                     .format(line))
    if len(vals) % 2 != 0:
        raise ValueError("anchors文件数值个数({0})必须为偶数（宽高成对）"
                         .format(len(vals)))
    anchors = np.array(vals, dtype=np.float32).reshape(-1, 2).tolist()
    return anchors


def random_generate_colors(color_num):
    """
    这是随机生成RGB颜色数组的函数
    Args:
        color_num: 颜色个数
    Returns:
    """
    hsv_tuples = [(x / color_num, 1., 1.)
                  for x in range(color_num)]
    rgb_colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
    rgb_colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), rgb_colors))
    return rgb_colors

def draw_detection_results(image,detecion_outputs,colors):
    """
    这是绘制检测结果的函数
    Args:
        image: 图像，opnecv读入
        detecion_outputs: 检测结果字典数组
        colors: rgb颜色列表
    Returns:
    """
    h,w,_= np.shape(image)
    tl = max(round((h + w) // 300),1)  # line/font thickness
    for output in detecion_outputs:
        x1,y1,x2,y2 = output['bbox']
        score = output['score']
        cls_name = output['cls_name']
        cls_id = output['cls_id']
        x1 = int(round(x1))
        y1 = int(round(y1))
        x2 = int(round(x2))
        y2 = int(round(y2))
        text = "{0}:{1:.4f}".format(cls_name,score)
        # 绘制检测框
        cv2.rectangle(image, (x1, y1), (x2, y2), colors[cls_id], thickness=tl, lineType=cv2.LINE_AA)
        # 初始化标签字符串宽高及其坐标
        tf = max(tl - 1, 1)
        text_w,text_h = cv2.getTextSize(text, 0, fontScale=tl / 3, thickness=tf)[0]
        outside = y1 - text_h >= 3
        text_x1,text_y1 = x1,y1
        text_x2,text_y2 = x1+text_w,y1-text_h-3 if outside else y1+text_h+3
        mean_color = np.mean(colors[cls_id])
        if mean_color > 128:
            text_color = (0, 0, 0)
        else:
            text_color = (255, 255, 255)
        # 绘制标签字符串
        cv2.rectangle(image, (text_x1,text_y1), (text_x2,text_y2),
                      colors[cls_id],thickness=-1, lineType=cv2.LINE_AA)
        cv2.putText(image, text, (x1,y1-2 if outside else y1+text_h+2),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=tl / 3,
                    color=text_color, thickness=tf, lineType=cv2.LINE_AA)
    return image

def x1y1x2y2_to_x1y1wh(box):
    """
    这是检测框格式转换函数：x1y1x2y2 --> x1y1wh
    Args:
        box: 检测框，格式为[x1,y1,x2,y2]
    Returns:
    """
    x1,y1,x2,y2 = box
    w = x2 - x1
    h = y2 - y1
    return [x1,y1,w,h]
