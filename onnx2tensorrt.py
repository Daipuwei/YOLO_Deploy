# -*- coding: utf-8 -*-
# @Time    : 2024/1/15 下午9:39
# @Author  : DaiPuWei
# @Email   : daipuwei@qq.com
# @File    : onnx2tensorrt.py
# @Software: PyCharm

"""
    这是ONNX转TensorRT模型的脚本
"""

import os

from utils import ArgsParser
from utils import logger_config
from engine import onnx2tensorrt
from engine import TensorRT_Calibrator

parser = ArgsParser()
parser.add_argument('--onnx_model_path', type=str, default='./model_data/v2x_yolov6s.onnx', help='onnx model path')
parser.add_argument('--tensorrt_model_path', type=str, default='./model_data/v2x_yolov6s.trt', help='tensorrt model path')
parser.add_argument('--input_shape', type=int, nargs='+',help='input shape')
parser.add_argument('--model_type', type=str, default="yolov5", help='model type: [yolov6,ppclas, pulc]')
parser.add_argument('--mode', type=str, default='test', help='tensorrt model mode: [fp32, fp16, int8]')
parser.add_argument('--calibrator_image_dir', type=str, default='', help='calibrator imageset directory')
parser.add_argument('--calibrator_table_path', type=str, default='', help='calibrator table path')
parser.add_argument('--data_type', type=str, default='float32', help='data type')
opt = parser.parse_args()

def ONNX2TensorRT(opt):
    """
    这是ONNX转TensorRT的函数
    Args:
        opt: 命令行解析类实例
    Returns:
    """
    # 初始化模型相关变量
    mode = opt.mode
    model_type = opt.model_type
    input_shape = opt.input_shape
    onnx_model_path = os.path.abspath(opt.onnx_model_path)
    tensorrt_model_path = os.path.abspath("{}.{}".format(opt.tensorrt_model_path,mode))
    calibrator_image_dir = os.path.abspath(opt.calibrator_image_dir)
    calibrator_table_path = os.path.abspath(opt.calibrator_table_path)
    data_type = opt.data_type
    logger = logger_config("./log.log","ONNX2TensorRT")
    if mode == 'int8':
        if model_type == 'yolov5':
            from model.yolov5 import YOLOv5_Calibration_Dataloader
            calibration_dataloader = YOLOv5_Calibration_Dataloader(logger,input_shape,calibrator_image_dir,data_type)
            calibrator = TensorRT_Calibrator(calibration_dataloader,calibrator_table_path)
        else:
            from model.yolov5 import YOLOv5_Calibration_Dataloader
            calibration_dataloader = YOLOv5_Calibration_Dataloader(logger, input_shape, calibrator_image_dir, data_type)
            calibrator = TensorRT_Calibrator(calibration_dataloader, calibrator_table_path)
    else:
        calibrator = None

    # 解析ONNX生成TensorRT模型
    onnx2tensorrt(logger,onnx_model_path,tensorrt_model_path,mode,calibrator)

def run_main():
    """
    这是主函数
    """
    ONNX2TensorRT(opt)

if __name__ == '__main__':
    run_main()
