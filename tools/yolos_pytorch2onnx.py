# -*- coding: utf-8 -*-
# @Time    : 2024/3/10 下午8:39
# @Author  : DaiPuWei
# @Email   : daipuwei@qq.com
# @File    : pytorch2onnx.py
# @Software: PyCharm

import argparse
import warnings
from pathlib import Path

import numpy as np
import torch
from models import build_model as build_yolos_model

import onnxsim
import onnx
import os


def get_args_parser():
    parser = argparse.ArgumentParser('Set YOLOS', add_help=False)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--lr_backbone', default=1e-5, type=float)
    parser.add_argument('--batch_size', default=2, type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=150, type=int)
    parser.add_argument('--eval_size', default=800, type=int)

    parser.add_argument('--clip_max_norm', default=0.1, type=float,
                        help='gradient clipping max norm')

    parser.add_argument('--use_checkpoint', action='store_true',
                        help='use checkpoint.checkpoint to save mem')
    # scheduler
    # Learning rate schedule parameters
    parser.add_argument('--sched', default='warmupcos', type=str, metavar='SCHEDULER',
                        help='LR scheduler (default: "step", options:"step", "warmupcos"')
    ## step
    parser.add_argument('--lr_drop', default=100, type=int)
    ## warmupcosine

    # parser.add_argument('--lr-noise', type=float, nargs='+', default=None, metavar='pct, pct',
    #                     help='learning rate noise on/off epoch percentages')
    # parser.add_argument('--lr-noise-pct', type=float, default=0.67, metavar='PERCENT',
    #                     help='learning rate noise limit percent (default: 0.67)')
    # parser.add_argument('--lr-noise-std', type=float, default=1.0, metavar='STDDEV',
    #                     help='learning rate noise std-dev (default: 1.0)')
    parser.add_argument('--warmup-lr', type=float, default=1e-6, metavar='LR',
                        help='warmup learning rate (default: 1e-6)')
    parser.add_argument('--min-lr', type=float, default=1e-7, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')
    parser.add_argument('--warmup-epochs', type=int, default=0, metavar='N',
                        help='epochs to warmup LR, if scheduler supports')
    parser.add_argument('--decay-rate', '--dr', type=float, default=0.1, metavar='RATE',
                        help='LR decay rate (default: 0.1)')

    # * model setting
    parser.add_argument("--det_token_num", default=100, type=int,
                        help="Number of det token in the deit backbone")
    parser.add_argument('--backbone_name', default='tiny', type=str,
                        help="Name of the deit backbone to use")
    parser.add_argument('--pre_trained', default='',
                        help="set imagenet pretrained model path if not train yolos from scatch")
    parser.add_argument('--init_pe_size', nargs='+', type=int,
                        help="init pe size (h,w)")
    parser.add_argument('--mid_pe_size', nargs='+', type=int,
                        help="mid pe size (h,w)")
    # * Matcher
    parser.add_argument('--set_cost_class', default=1, type=float,
                        help="Class coefficient in the matching cost")
    parser.add_argument('--set_cost_bbox', default=5, type=float,
                        help="L1 box coefficient in the matching cost")
    parser.add_argument('--set_cost_giou', default=2, type=float,
                        help="giou box coefficient in the matching cost")
    # * Loss coefficients

    parser.add_argument('--dice_loss_coef', default=1, type=float)
    parser.add_argument('--bbox_loss_coef', default=5, type=float)
    parser.add_argument('--giou_loss_coef', default=2, type=float)
    parser.add_argument('--eos_coef', default=0.1, type=float,
                        help="Relative classification weight of the no-object class")

    # dataset parameters
    parser.add_argument('--dataset_file', default='coco')
    parser.add_argument('--coco_path', type=str)
    parser.add_argument('--coco_panoptic_path', type=str)
    parser.add_argument('--remove_difficult', action='store_true')

    parser.add_argument('--output_dir', default='',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--num_workers', default=2, type=int)

    # onnx
    parser.add_argument('--onnx_model_path', default='', type=str, help='onnx model path')
    parser.add_argument('--opset', default=11, type=int, help='opset version')
    parser.add_argument('--height', default=1333, type=int, help='image height')
    parser.add_argument('--width', default=800, type=int, help='image width')

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    return parser


def main(args):
    # 初始化模型
    device = torch.device(args.device)
    model, criterion, postprocessors = build_yolos_model(args)
    # model, criterion, postprocessors = build_model(args)
    model.to(device)

    # 加载模型
    checkpoint = torch.load(args.resume, map_location=device)
    model.load_state_dict(checkpoint['model'])
    print("Loadin weight from ",args.resume)

    # onnx转换
    onnx_model_path = args.onnx_model_path
    dir,onnx_model_name = os.path.split(onnx_model_path)
    fname,ext = os.path.splitext(onnx_model_name)
    onnx_model_path = os.path.join(dir,"{0}_{1}x{2}{3}".format(fname,args.height,args.width,ext))
    input_tensor = torch.from_numpy(np.random.randn(1,3,args.height,args.width).astype(np.float32)).to(device)
    model.forward = model.forward_dummy
    torch.onnx.export(
        model,
        input_tensor,
        onnx_model_path,
        input_names=['input'],
        output_names=['output'],
        export_params=True,
        keep_initializers_as_inputs=True,
        do_constant_folding=True,
        verbose=True,
        opset_version=args.opset)

    model_opt, check_ok = onnxsim.simplify(onnx_model_path)
    if check_ok:
        onnx.save(model_opt,onnx_model_path)
        print(f'Successfully simplified ONNX model: {onnx_model_path}')
    else:
        warnings.warn('Failed to simplify ONNX model.')
    print(f'Successfully exported ONNX model: {onnx_model_path}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser('YOLOS training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)