# -*- coding: utf-8 -*-
# @Time    : 2023/4/2 下午4:47
# @Author  : DaiPuWei
# @Email   : daipuwei@qq.com
# @File    : compute_mAP_COCO.py
# @Software: PyCharm

"""
    这是利用COCO API计算mAP的脚本
"""

import os
import json
import argparse
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

parser = argparse.ArgumentParser()
parser.add_argument('--gt_json_path', type=str, default='./VOC2007', help='groundtruth json path')
parser.add_argument('--dr_json_path', type=str, default='./VOC2007-COCO', help='detection result json path')
opt = parser.parse_args()

def get_img_id(json_path):
    """
    这是从JSON文件中获取图像id的函数
    Args:
        json_path: json文件路径
    Returns:
    """
    ls = []
    myset = []
    annos = json.load(open(json_path,'r'))
    for anno in annos['annotations']:
        #print(anno)
        #print(anno['image_id'])
        ls.append(anno['image_id'])
    myset = {}.fromkeys(ls).keys()
    return myset

def run_main():
    """
    这是主函数
    """
    gt_json_path = os.path.abspath(opt.gt_json_path)
    dr_json_path = os.path.abspath(opt.dr_json_path)

    coco_gt = COCO(gt_json_path)
    coco_dr = COCO(dr_json_path)
    img_ids = get_img_id(dr_json_path)
    img_ids = sorted(img_ids)
    coco_eval = COCOeval(coco_gt,coco_dr,'bbox')
    coco_eval.params.imgIds = img_ids
    coco_eval.evaluate()#评价
    coco_eval.accumulate()#积累
    coco_eval.summarize()#总结

if __name__ == '__main__':
    run_main()
