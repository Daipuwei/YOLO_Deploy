# -*- coding: utf-8 -*-
# @Time    : 2023/4/2 下午6:08
# @Author  : DaiPuWei
# @Email   : daipuwei@qq.com
# @File    : test.py
# @Software: PyCharm

"""
    这是在VOC格式的数据集上测试模型性能的脚本
"""

import os
import cv2
import json
import argparse
import numpy as np
from tqdm import tqdm
from datetime import datetime
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from utils import load_yaml
from utils import NpEncoder
from utils import VOCDataset
from utils import COCODataset
from utils import logger_config
from utils import draw_detection_results

parser = argparse.ArgumentParser()
parser.add_argument('--cfg', type=str, default='./config/detection.yaml', help='config yaml file path')
parser.add_argument('--dataset_dir', type=str, default='./VOC2007', help='dataset directory')
parser.add_argument('--dataset_type', type=str, default="voc",help='dataset type: voc or coco')
parser.add_argument('--choice', type=str, default="val",help='VOC dataset choice')
parser.add_argument('--result_dir', type=str, default="./result/eval/VOC2007", help='result save directory')
parser.add_argument('--iou_threshold', type=float, default=0.5,help='iou threshold for nms')
parser.add_argument('--confidence_threshold', type=float, default=0.001,help='confidence threshold')
parser.add_argument('--save_image', action='store_true', help='save detection image')
opt = parser.parse_args()

def init_cfg(opt):
    """
    这是初始化配置参数字典的函数
    Args:
        opt: 参数字典
    Returns:
    """
    cfg = load_yaml(opt.cfg)
    cfg["DetectionModel"]['confidence_threshold'] = opt.confidence_threshold
    cfg["DetectionModel"]['iou_threshold'] = opt.iou_threshold
    return cfg
def test(logger,detection_model,dataset_dir,result_dir,dataset_type='voc',mode='val',save_image=False):
    """
    这是对模型性能进行测试测试的函数
    Args:
        logger: 日志类实例
        detection_model: 检测模型实例
        dataset_dir: 数据集地址
        result_dir: 结果文件夹地址
        dataset_type: 数据集类型，默认为'voc',候选值为['voc','coco']
        mode：VOC数据集类型，默认为‘val’
        save_image: 是否保存检测结果图像标志量，默认为False
    Returns:
    """
    # 初始化相关文件路径
    logger.info("开始加载检测数据集")
    batch_size = detection_model.get_batch_size()
    if dataset_type == 'voc':
        test_dataloader = VOCDataset(dataset_dir,batch_size,mode=mode)
    else:
        test_dataloader = COCODataset(dataset_dir, batch_size, mode=mode)
    image_num = test_dataloader.get_image_num()
    logger.info("结束加载检测数据集")
    logger.info("共计有{}张图像需要进行检测".format(image_num))

    # 初始化json文件路径
    time = datetime.now().strftime('%Y%m%d%H%M%S')
    result_dir = os.path.join(result_dir, mode, time)
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    result_txt_path = os.path.join(result_dir, 'result.txt')
    pred_json_result_path = os.path.join(result_dir, 'detection_result.json')
    gt_json_result_path = os.path.join(result_dir, 'gt_result.json')
    detect_image_dir = os.path.join(result_dir, 'images')
    if not os.path.exists(detect_image_dir):
        os.makedirs(detect_image_dir)
    detection_results = []
    image_infos = []
    gt_results = []

    # 检测图像
    detection_model_time_dict = {"preprocess_time": [],
                                 "inference_time": [],
                                 "postprocess_time": [],
                                 "detect_time": []}
    class_names = detection_model.get_class_names()
    colors = detection_model.get_colors()
    image_cnt = 0
    gt_annotation_cnt = 0
    pred_annotation_cnt = 0
    for batch_images, batch_gts, batch_image_paths in tqdm(test_dataloader):
        pred_results = detection_model.detect(batch_images,True)
        if len(pred_results) == 2:
            batch_preds, _detection_model_time_dict = pred_results
            if batch_size == 1:
                batch_preds = [batch_preds]
                detection_model_time_dict['preprocess_time'].append(_detection_model_time_dict["preprocess_time"])
                detection_model_time_dict['inference_time'].append(_detection_model_time_dict["inference_time"])
                detection_model_time_dict['postprocess_time'].append(_detection_model_time_dict["postprocess_time"])
                detection_model_time_dict['detect_time'].append(_detection_model_time_dict["detect_time"])
                # 遍历每张图像的检测结果，并评估性能
                for image, gts, preds, image_path in zip(batch_images, batch_gts, batch_preds, batch_image_paths):
                    # 保存检测图片
                    if save_image:
                        _, image_name = os.path.split(image_path)
                        draw_image = draw_detection_results(image, preds, class_names, colors)
                        cv2.imwrite(os.path.join(detect_image_dir, image_name), draw_image)
                    # 初始化图像信息
                    h, w, c = np.shape(image)
                    _, image_name = os.path.split(image_path)
                    image_infos.append({'file_name': image_name, 'id': image_cnt, 'width': w, 'height': h})
                    # 初始化图像每个gt信息
                    for obj_name,x1,y1,bbox_w,bbox_h in gts:
                        gt_results.append({'image_id': image_cnt,
                                           'iscrowd': 0,
                                           "bbox": [x1,y1,bbox_w,bbox_h],
                                           'area': bbox_w*bbox_h,
                                           "category_id": class_names.index(obj_name),
                                           'id': gt_annotation_cnt})
                        gt_annotation_cnt += 1
                    # 初始化图像中每个预测结果
                    for x1, y1, x2, y2, score, cls_id in preds:
                        bbox_w = x2 - x1
                        bbox_h = y2 - y1
                        cls_id = int(cls_id)
                        detection_results.append({'image_id': image_cnt,
                                                  'iscrowd': 0,
                                                  'category_id': cls_id,
                                                  "bbox": [x1, y1, bbox_w, bbox_h],
                                                  'area': bbox_w * bbox_h,
                                                  'id': pred_annotation_cnt,
                                                  'score': score})
                        pred_annotation_cnt += 1
                    image_cnt += 1

    # 计算检测时间
    detection_model_time_dict['preprocess_time'] = np.mean(detection_model_time_dict["preprocess_time"])
    detection_model_time_dict['inference_time'] = np.mean(detection_model_time_dict["inference_time"])
    detection_model_time_dict['postprocess_time'] = np.mean(detection_model_time_dict["postprocess_time"])
    detection_model_time_dict['detect_time'] = np.mean(detection_model_time_dict["detect_time"])
    logger.info("batchsize={0},平均预处理时间为：{1:.4f}ms".format(batch_size,detection_model_time_dict['preprocess_time']))
    logger.info("batchsize={0},平均推理时间为：{1:.4f}ms".format(batch_size,detection_model_time_dict['inference_time']))
    logger.info("batchsize={0},平均后处理时间为：{1:.4f}ms".format(batch_size,detection_model_time_dict['postprocess_time']))
    logger.info("batchsize={0},平均检测时间为：{1:.4f}ms".format(batch_size,detection_model_time_dict['detect_time']))

    # 保存预测和gt
    logger.info("测试集路径为：{0}".format(dataset_dir))
    with open(gt_json_result_path, 'w+', encoding="utf-8") as f:
        gt_json_dict = {}
        gt_json_dict['images'] = image_infos
        gt_json_dict["annotations"] = gt_results
        gt_json_dict["categories"] = [{"id": id, "name": cls_name} for id, cls_name in enumerate(class_names)]
        json_data = json.dumps(gt_json_dict, indent=4, separators=(',', ': '),
                               cls=NpEncoder, ensure_ascii=False)
        f.write(json_data)
    logger.info("测试集真实标签保存路径为：{0}".format(gt_json_result_path))
    with open(pred_json_result_path, 'w+', encoding="utf-8") as f:
        # json_data = json.dumps(detection_results, indent=4, separators=(',', ': '),
        #                        cls=NpEncoder, ensure_ascii=False)
        pred_json_dict = {}
        pred_json_dict['images'] = image_infos
        pred_json_dict["annotations"] = np.array(detection_results)
        pred_json_dict["categories"] = [{"id": id, "name": cls_name} for id, cls_name in enumerate(class_names)]
        json_data = json.dumps(pred_json_dict, indent=4, separators=(',', ': '),
                               cls=NpEncoder, ensure_ascii=False)
        f.write(json_data)
    logger.info("测试集检测标签保存路径为：{0}".format(pred_json_result_path))

    # 计算车牌检测模型测试集上的性能
    cocoGt = COCO(gt_json_result_path)
    #cocoDt = cocoGt.loadRes(pred_json_result_path)
    cocoDt = COCO(pred_json_result_path)

    # 创建检测框评估器
    evaluator = COCOeval(cocoGt, cocoDt,iouType='bbox')
    evaluator.evaluate()
    evaluator.accumulate()
    evaluator.summarize()
    map, map50 = evaluator.stats[:2]  # update results (mAP@0.5:0.95, mAP@0.5)
    mar = evaluator.stats[8]

    # 计算每个类别的AP和AR指标，iou=0.5
    class_stats = []
    eval_info_list = ["{0}\t{1}\t{2}".format("category name".rjust(15),"mAP@0.5".rjust(15),"mAR@0.5".rjust(15))]
    for i in range(len(class_names)):
        stats, print_info = summarize(evaluator, catId=i)
        ap50,ap75,ap95,ar50,ar75,ar95 =stats[1:7]
        class_stats.append([ap50,ap75,ap95,ar50,ar75,ar95])
        eval_info_list.append("{0}\t{1:15.4f}\t{2:15.4f}".format(class_names[i].rjust(15),ap50,ar50))
    class_stats = np.array(class_stats)
    class_stats = np.mean(class_stats, 0)
    eval_info = "\n".join(eval_info_list)

    # 打印模型在测试集上的检测性能
    logger.info("检测模型在测试集上mAP@0.5={0:.4f},mAP@0.5:0.95={1:.4f}".format(map50, map))
    logger.info("检测模型在测试集上mAP@0.5={0:.4f},mAR@0.5={1:.4f}".format(class_stats[0], class_stats[1]))
    logger.info("检测模型在测试集上mAP@0.5:0.95={0:.4f},mAR@0.5:0.95={1:.4f}".format(map, mar))
    logger.info("检测模型在测试集上各个类别检测性能如下：")
    logger.info(eval_info)

    # 保存结果到txt
    with open(result_txt_path, 'w+', encoding="utf-8") as f:
        f.write("batchsize={0},平均预处理时间为：{1:.4f}ms\n"
                .format(batch_size,detection_model_time_dict['preprocess_time']))
        f.write("batchsize={0},平均推理时间为：{1:.4f}ms\n"
                .format(batch_size, detection_model_time_dict['inference_time']))
        f.write("batchsize={0},平均后处理时间为：{1:.4f}ms\n"
                .format(batch_size,detection_model_time_dict['postprocess_time']))
        f.write("batchsize={0},平均检测时间为：{1:.4f}ms\n"
                .format(batch_size, detection_model_time_dict['detect_time']))
        f.write("测试集路径为：{0}\n".format(dataset_dir))
        f.write("测试集真实标签保存路径为：{0}\n".format(gt_json_result_path))
        f.write("测试集检测标签保存路径为：{0}\n".format(pred_json_result_path))
        f.write("检测模型在测试集上mAP@0.5={0:.4f},mAP@0.5:0.95={1:.4f}\n".format(map50,map))
        f.write("检测模型在测试集上mAP@0.5={0:.4f},mAR@0.5={1:.4f}\n".format(class_stats[0],class_stats[1]))
        f.write("检测模型在测试集上mAP@0.5:0.95={0:.4f},mAR@0.5:0.95={1:.4f}\n".format(map,mar))
        f.write("检测模型在测试集上各个类别检测性能如下：\n")
        f.write(eval_info)

def summarize(self, catId=None):
    """
    Compute and display summary metrics for evaluation results.
    Note this functin can *only* be applied on the default parameter setting
    """

    def _summarize(ap=1, iouThr=None, areaRng='all', maxDets=100):
        p = self.params
        iStr = ' {:<18} {} @[ IoU={:<9} | area={:>6s} | maxDets={:>3d} ] = {:0.3f}'
        titleStr = 'Average Precision' if ap == 1 else 'Average Recall'
        typeStr = '(AP)' if ap == 1 else '(AR)'
        iouStr = '{:0.2f}:{:0.2f}'.format(p.iouThrs[0], p.iouThrs[-1]) \
            if iouThr is None else '{:0.2f}'.format(iouThr)

        aind = [i for i, aRng in enumerate(p.areaRngLbl) if aRng == areaRng]
        mind = [i for i, mDet in enumerate(p.maxDets) if mDet == maxDets]

        if ap == 1:
            # dimension of precision: [TxRxKxAxM]
            s = self.eval['precision']
            # IoU
            if iouThr is not None:
                t = np.where(iouThr == p.iouThrs)[0]
                s = s[t]

            if isinstance(catId, int):
                s = s[:, :, catId, aind, mind]
            else:
                s = s[:, :, :, aind, mind]

        else:
            # dimension of recall: [TxKxAxM]
            s = self.eval['recall']
            if iouThr is not None:
                t = np.where(iouThr == p.iouThrs)[0]
                s = s[t]

            if isinstance(catId, int):
                s = s[:, catId, aind, mind]
            else:
                s = s[:, :, aind, mind]

        if len(s[s > -1]) == 0:
            mean_s = -1
        else:
            mean_s = np.mean(s[s > -1])

        print_string = iStr.format(titleStr, typeStr, iouStr, areaRng, maxDets, mean_s)
        return mean_s, print_string

    stats, print_list = [0] * 10, [""] * 10
    stats[0], print_list[0] = _summarize(1)
    stats[1], print_list[1] = _summarize(1, iouThr=.5, maxDets=self.params.maxDets[2])
    stats[2], print_list[2] = _summarize(1, iouThr=.75, maxDets=self.params.maxDets[2])
    stats[3], print_list[3] = _summarize(1, iouThr=.95, maxDets=self.params.maxDets[2])
    stats[4], print_list[4] = _summarize(0, iouThr=.5, maxDets=self.params.maxDets[2])
    stats[5], print_list[5] = _summarize(0, iouThr=.75, maxDets=self.params.maxDets[2])
    stats[6], print_list[6] = _summarize(0, iouThr=.95, maxDets=self.params.maxDets[2])
    stats[7], print_list[7] = _summarize(0, maxDets=self.params.maxDets[0])
    stats[8], print_list[8] = _summarize(0, maxDets=self.params.maxDets[1])
    stats[9], print_list[9] = _summarize(0, maxDets=self.params.maxDets[2])

    print_info = "\n".join(print_list)

    if not self.eval:
        raise Exception('Please run accumulate() first')

    return stats, print_info

def run_main():
    """
    这是主函数
    """
    # 初始化而皮脂参数字典
    cfg = init_cfg(opt)

    # 初始化检测模型
    model_type = cfg["DetectionModel"]["model_type"].lower()
    logger = logger_config(cfg['log_path'], model_type)
    if model_type == 'yolov5':
        from model import YOLOv5
        detection_model = YOLOv5(logger=logger, cfg=cfg)
    else:
        from model import YOLOv5
        detection_model = YOLOv5(logger=logger, cfg=cfg)

    # 初始化相关路径路径
    result_dir = os.path.abspath(opt.result_dir)
    dataset_dir = os.path.abspath(opt.dataset_dir)

    # 对检测模型记性预热，tensorRT等模型前几次推理时间较长，影响评测结果
    logger.info("预热模型开始")
    if cfg['DetectionModel']['input_shape'][1] > 3:
        batch_size, h, w, c = cfg['DetectionModel']['input_shape']
    else:
        batch_size, c, h, w, = cfg['DetectionModel']['input_shape']
    image_tensor = np.random.random((batch_size,h,w,c))
    for i in np.arange(100):
        detection_model.detect(image_tensor)
    logger.info("预热模型结束")

    # 测试检测性能
    test(logger,detection_model,dataset_dir,result_dir,opt.dataset_type,opt.choice,opt.save_image)

if __name__ == '__main__':
    run_main()
