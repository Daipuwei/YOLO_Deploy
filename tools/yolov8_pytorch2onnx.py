# -*- coding: utf-8 -*-
# @Time    : 2024/3/18 下午4:01
# @Author  : DaiPuWei
# @Email   : daipuwei@qq.com
# @File    : pytorch2onnx.py
# @Software: PyCharm

from ultralytics import YOLO
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument('--model_path', type=str, default='yolov8n.pt', help='yolo model path')
parser.add_argument('--simplify', action='store_true', help='simplify onnx')
parser.add_argument("--imgsz", "--img", "--img-size", nargs="+", type=int, default=[640], help="inference size h,w")
parser.add_argument('--opset', type=int, default=11, help='opset version')
opt = parser.parse_args()

def run_main():
    """
    这是主函数
    """
    # Load a model
    model = YOLO(opt.model_path)  # load an official model

    # Export the model
    model.export(format='onnx', simplify=opt.simplify,opset=opt.opset,imgsz=opt.imgsz)


if __name__ == '__main__':
    run_main()
