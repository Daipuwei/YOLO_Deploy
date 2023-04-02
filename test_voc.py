# -*- coding: utf-8 -*-
# @Time    : 2023/4/2 下午6:08
# @Author  : DaiPuWei
# @Email   : daipuwei@qq.com
# @File    : test_voc.py
# @Software: PyCharm

"""
    这是在COCO数据集上测试模型性能的脚本
"""

import os
import cv2
import argparse
import numpy as np
from tqdm import tqdm

from utils import load_yaml
from utils import logger_config
from utils import draw_detection_results

parser = argparse.ArgumentParser()
parser.add_argument('--cfg', type=str, default='./config/detection.yaml', help='config yaml file path')
parser.add_argument('--voc_dataset_dir', type=str, default='./VOC2007', help='VOC dataset directory')
parser.add_argument('--result_dir', type=str, default="./mAP/input/VOC2007", help='result save directory')
parser.add_argument('--choice', type=str, default="val",help='VOC dataset choice')
parser.add_argument('--model_name', type=str, default="yolov5s",help="detection model name")
parser.add_argument('--save_image', action='store_true', help='save detection image')
opt = parser.parse_args()

def test(logger,detection_model,voc_dataset_dir,result_dir,choice='val',save_image=False):
    """
    这是对模型性能进行测试测试的函数
    Args:
        logger: 日志类实例
        detection_model: 检测模型实例
        voc_dataset_dir: VOC数据集地址
        result_dir: 结果文件夹地址
        choice：VOC评测子集名称，默认为‘val’
        save_image: 是否保存检测结果图像标志量，默认为False
    Returns:
    """
    # 初始化相关文件路径
    logger.info("开始初始化检测图像集路径")
    result_image_dir = os.path.join(result_dir,'images')
    result_detection_dir = os.path.join(result_dir,'detection-results')
    if not os.path.exists(result_image_dir):
        os.makedirs(result_image_dir)
    if not os.path.exists(result_detection_dir):
        os.makedirs(result_detection_dir)
    voc_image_paths = []
    result_image_paths = []
    result_annotation_txt_paths = []
    voc_image_dir = os.path.join(voc_dataset_dir,"JPEGImages")
    voc_main_dir = os.path.join(voc_dataset_dir,"ImageSets","Main")
    voc_txt_path = os.path.join(voc_main_dir,choice+".txt")
    with open(voc_txt_path,'r') as f:
        for line in f.readlines():
            voc_image_paths.append(os.path.join(voc_image_dir,line.strip()+".jpg"))
            result_image_paths.append(os.path.join(result_image_dir,line.strip()+".jpg"))
            result_annotation_txt_paths.append(os.path.join(result_detection_dir,line.strip()+".txt"))
    voc_image_paths = np.array(voc_image_paths)
    result_image_paths = np.array(result_image_paths)
    result_annotation_txt_paths = np.array(result_annotation_txt_paths)
    logger.info("结束初始化检测图像路径")
    logger.info("共计有{}张图像需要进行检测".format(len(voc_image_paths)))

    # 检测图像
    size = len(voc_image_paths)
    batch_size = detection_model.get_batch_size()
    class_names = detection_model.get_class_names()
    preprocess_times = []
    inference_times = []
    postprocess_times = []
    detect_times = []
    for start in tqdm(range(0, size, batch_size)):
        images = []
        end = int(np.min([start + batch_size, size]))
        _voc_image_paths = voc_image_paths[start:end]
        _result_image_paths = result_image_paths[start:end]
        _result_annotation_paths = result_annotation_txt_paths[start:end]
        for voc_image_path in _voc_image_paths:
            image = cv2.imread(voc_image_path)
            images.append(image)
        preds,preprocess_time,inference_time,postprocess_time,detect_time = detection_model.detect(images,True)
        if batch_size == 1:
            preds = [preds]
        preprocess_times.append(preprocess_time)
        inference_times.append(inference_time)
        postprocess_times.append(postprocess_time)
        detect_times.append(detect_time)
        for i, (voc_image_path, result_image_path, result_annotation_path) in \
                enumerate(zip(_voc_image_paths, _result_image_paths, _result_annotation_paths)):
            # 绘制检测结果
            if save_image:
                drawed_image = draw_detection_results(images[i], preds[i])
                cv2.imwrite(result_image_path, drawed_image)
            if len(preds[i]) > 0:
                for x1, y1, x2, y2, score, cls_id in preds[i]:
                    x1 = int(round(x1))
                    y1 = int(round(y1))
                    x2 = int(round(x2))
                    y2 = int(round(y2))
                    cls_id = int(cls_id)
                    with open(result_annotation_path, 'w') as f:
                        f.write("{0} {1} {2} {3} {4} {5}\n".format(class_names[cls_id],score,x1, y1, x2, y2))

    # 计算检测时间
    logger.info("平均预处理时间为：{0:.2f}ms".format(round(np.mean(preprocess_times),2)))
    logger.info("平均推理时间为：{0:.2f}ms".format(round(np.mean(inference_times),2)))
    logger.info("平均后处理时间为：{0:.2f}ms".format(round(np.mean(postprocess_times),2)))
    logger.info("平均检测时间为：{0:.2f}ms".format(round(np.mean(detect_times),2)))

def run_main():
    """
    这是主函数
    """
    # 初始化检测模型
    cfg = load_yaml(opt.cfg)
    model_type = cfg["DetectionModel"]["model_type"].lower()
    logger = logger_config(cfg['log_path'], model_type)
    if model_type == 'yolov5':
        from model import YOLOv5
        detection_model = YOLOv5(logger=logger, cfg=cfg)
    else:
        from model import YOLOv5
        detection_model = YOLOv5(logger=logger, cfg=cfg)

    # 初始化相关路径路径
    result_dir = os.path.join(opt.result_dir,opt.model_name)
    voc_dataset_dir = os.path.abspath(opt.voc_dataset_dir)

    # 对检测模型记性预热，tensorRT等模型前几次推理时间较长，影响评测结果
    logger.info("预热模型开始")
    batchsize,c,h,w = cfg['DetectionModel']['input_shape']
    image_tensor = np.random.random((batchsize,h,w,c))
    for i in np.arange(100):
        detection_model.detect(image_tensor)
    logger.info("预热模型结束")

    # 测试检测性能
    test(logger,detection_model,voc_dataset_dir,result_dir,opt.choice,opt.save_image)

if __name__ == '__main__':
    run_main()