# -*- coding: utf-8 -*-
# @Time    : 2023/4/2 上午1:06
# @Author  : DaiPuWei
# @Email   : daipuwei@qq.com
# @File    : test_coco.py
# @Software: PyCharm

"""
    这是在COCO数据集上测试模型性能的脚本
"""

import os
import cv2
import json
import argparse
import numpy as np
from tqdm import tqdm

from utils import NpEncoder
from utils import load_yaml
from utils import logger_config
from utils import draw_detection_results

IMG_FORMATS = ['.bmp', '.dng', '.jpeg', '.jpg', '.mpo', '.png', '.tif', '.tiff', '.webp']  # include image suffixes

parser = argparse.ArgumentParser()
parser.add_argument('--cfg', type=str, default='./config/detection.yaml', help='config yaml file path')
parser.add_argument('--coco_image_dir', type=str, default='./images', help='coco image directory')
parser.add_argument('--result_dir', type=str, default="./result", help='result save directory')
parser.add_argument('--model_name', type=str, default="yolov5s",help="detection model name")
parser.add_argument('--save_image', action='store_true', help='save detection image')
opt = parser.parse_args()

def test(logger,detection_model,image_dir,result_dir,save_image=False):
    """
    这是对模型性能进行测试测试的函数
    Args:
        logger: 日志类实例
        detection_model: 检测模型实例
        image_dir: 图像文件夹地址
        result_dir: 结果文件夹地址
        save_image: 是否保存检测结果图像标志量，默认为False
    Returns:
    """
    # 初始化相关文件路径
    logger.info("开始初始化检测图像路径")
    result_image_dir = os.path.join(result_dir,'images')
    result_detection_json_path = os.path.join(result_dir,'dr_result.json')
    if not os.path.exists(result_image_dir):
        os.makedirs(result_image_dir)
    coco_image_paths = []
    result_image_paths = []
    result_annotation_txt_paths = []
    image_ids = []
    # print(image_dir)
    for i,image_name in enumerate(os.listdir(image_dir)):
        fname,ext = os.path.splitext(image_name)
        if ext in IMG_FORMATS:
            coco_image_paths.append(os.path.join(image_dir,image_name))
            result_image_paths.append(os.path.join(result_image_dir,image_name))
            image_ids.append(i)
    coco_image_paths = np.array(coco_image_paths)
    result_image_paths = np.array(result_image_paths)
    result_annotation_txt_paths = np.array(result_annotation_txt_paths)
    image_ids = np.array(image_ids,dtype=np.int32)
    logger.info("结束初始化检测图像路径")
    logger.info("共计有{}张图像需要进行检测".format(len(coco_image_paths)))

    # 检测图像
    size = len(coco_image_paths)
    batch_size = detection_model.get_batch_size()
    class_names = detection_model.get_class_names()
    colors = detection_model.get_colors()
    dr_result = {}
    image_infos = []
    detection_results = []
    preprocess_times = []
    inference_times = []
    postprocess_times = []
    detect_times = []
    anno_cnt = 0
    for start in tqdm(range(0, size, batch_size)):
        images = []
        end = int(np.min([start + batch_size, size]))
        _coco_image_paths = coco_image_paths[start:end]
        _result_image_paths = result_image_paths[start:end]
        _result_annotation_paths = result_annotation_txt_paths[start:end]
        _image_ids = image_ids[start:end]
        for coco_image_path in _coco_image_paths:
            image = cv2.imread(coco_image_path)
            images.append(image)
        preds,preprocess_time,inference_time,postprocess_time,detect_time = detection_model.detect(images,True)
        #print(len(preds))
        if batch_size == 1:
            preds = [preds]
        preprocess_times.append(preprocess_time)
        inference_times.append(inference_time)
        postprocess_times.append(postprocess_time)
        detect_times.append(detect_time)
        for i in np.arange(len(_coco_image_paths)):
        # for i, (coco_image_path, image_id,result_image_path, result_annotation_path) in \
        #         enumerate(zip(_coco_image_paths, _image_ids,_result_image_paths, _result_annotation_paths)):
            image_h,image_w,c = np.shape(images[i])
            _,image_name = os.path.split(_coco_image_paths[i])
            image_infos.append({'file_name': image_name, 'id': _image_ids[i], 'width': image_w, 'height': image_h})
            # 绘制检测结果
            if save_image:
                drawed_image = draw_detection_results(images[i], preds[i],class_names,colors)
                cv2.imwrite(_result_image_paths[i], drawed_image)
            # print("=================")
            # print(len(preds[i]))
            # print("=================")
            if len(preds[i]) > 0:
                for x1, y1, x2, y2, score, cls_id in preds[i]:
                    w = round(x2 - x1)
                    h = round(y2 - y1)
                    cls_id = int(cls_id)
                    detection_results.append({'image_id': _image_ids[i],
                                              'iscrowd': 0,
                                              'category_id': cls_id,
                                              'bbox': [int(x1),int(y1),int(w),int(h)],
                                              'area': int(w * h),
                                              'id': anno_cnt,
                                              'score': score})
                    anno_cnt += 1

    # 计算检测时间
    logger.info("平均预处理时间为：{0}ms".format(np.mean(preprocess_times)))
    logger.info("平均推理时间为：{0}ms".format(np.mean(inference_times)))
    logger.info("平均后处理时间为：{0}ms".format(np.mean(postprocess_times)))
    logger.info("平均检测时间为：{0}ms".format(np.mean(detect_times)))

    # 检测结果写入JSON文件
    print(image_infos)
    print(detection_results)
    dr_result['images'] = image_infos
    dr_result["annotations"] = detection_results
    dr_result["categories"] = [{"id": id, "name": cls_name} for id,cls_name in enumerate(class_names)]
    dr_result_json_data = json.dumps(dr_result, indent=4, separators=(',', ': '), cls=NpEncoder)
    logger.info(dr_result_json_data)
    with open(result_detection_json_path, 'w+',encoding="utf-8") as f:
        f.write(dr_result_json_data)


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
    image_dir = os.path.abspath(opt.coco_image_dir)

    # 对检测模型记性预热，tensorRT等模型前几次推理时间较长，影响评测结果
    logger.info("预热模型开始")
    batchsize,c,h,w = cfg['DetectionModel']['input_shape']
    image_tensor = np.random.random((batchsize,h,w,c))
    for i in np.arange(100):
        detection_model.detect(image_tensor)
    logger.info("预热模型结束")

    # 测试检测性能
    test(logger,detection_model,image_dir,result_dir,opt.save_image)

if __name__ == '__main__':
    run_main()