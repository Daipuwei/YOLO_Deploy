# -*- coding: utf-8 -*-
# @Time    : 2023/4/2 下午5:51
# @Author  : DaiPuWei
# @Email   : daipuwei@qq.com
# @File    : get_gt_txt.py
# @Software: PyCharm

"""
    这是生成测试数据集每张图像中真实目标及其定位信息的脚本
"""

import os
import argparse
import xml.etree.ElementTree as ET

parser = argparse.ArgumentParser()
parser.add_argument('--input_dir', type=str, default='./VOC2007', help='gt and dr txt input directory')
parser.add_argument('--voc_dataset_dir', type=str, default='./VOC2007-COCO', help='VOC datset directory')
parser.add_argument('--choice', type=str, default='val', help='VOC dataset choice')
opt = parser.parse_args()

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
        left = bndbox.find('xmin').text
        top = bndbox.find('ymin').text
        right = bndbox.find('xmax').text
        bottom = bndbox.find('ymax').text
        objects.append("%s %s %s %s %s\n" % (obj_name, left, top, right, bottom))
    return objects

def run_main():
    """
       这是主函数
    """
    input_dir = os.path.abspath(opt.input_dir)
    groundtruth_dir = os.path.join(input_dir, "ground-truth")
    if not os.path.exists(groundtruth_dir):
        os.makedirs(groundtruth_dir)

    # 初始化测试数据集txt文件
    dataset_dir = os.path.abspath(opt.voc_dataset_dir)
    test_txt_path = os.path.join(dataset_dir, "ImageSets", "Main", opt.choice+".txt")
    image_ids = []
    with open(test_txt_path,'r') as f:
        for line in f.readlines():
            image_ids.append(line.strip())

    # 生成测试集的groundtruth的分类与定位的txt文件
    annotation_dir = os.path.join(dataset_dir,"Annotations")
    for image_id in image_ids:
        gt_txt_path = os.path.join(groundtruth_dir,image_id+".txt")
        with open(gt_txt_path, "w") as f:
            xml_path = os.path.join(annotation_dir,image_id+".xml")
            if is_contain_object(xml_path):
                objects = parse_xml(xml_path)
                for obj in objects:
                    f.write(obj)
    print("Test Dataset GroundTruth Result Conversion Completed!")

if __name__ == '__main__':
    run_main()
