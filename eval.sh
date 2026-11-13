#!/bin/bash
# 数据集性能评估入口（按需修改 --cfg/--dataset_dir/--dataset_type 与模型参数）
# --dataset_type 候选：voc / coco / yolo_v1 / yolo_v2

# 1) VOC 格式数据集（Annotations/JPEGImages/ImageSets）
python tools/eval.py --cfg ./config/detection.yaml \
                     --dataset_dir ./VOC2007 \
                     --dataset_type voc \
                     --choice val \
                     --result_dir ./result/eval \
                     --gpu_id 0 \
                     --metrics coco,yolo

# 2) COCO 格式数据集
python tools/eval.py --cfg ./config/detection.yaml \
                     --dataset_dir ./coco_dataset \
                     --dataset_type coco \
                     --choice val \
                     --result_dir ./result/eval \
                     --gpu_id 0 \
                     --metrics coco,yolo

# 3) YOLO v1 格式数据集（父目录含 images/、labels/ 与子集 txt 清单）
python tools/eval.py --cfg ./config/detection.yaml \
                     --dataset_dir ./yolo_dataset \
                     --dataset_type yolo_v1 \
                     --choice val \
                     --result_dir ./result/eval \
                     --gpu_id 0 \
                     --metrics coco,yolo

# 4) YOLO v2 格式数据集（父目录下每个子集文件夹含 images/、labels/）
python tools/eval.py --cfg ./config/detection.yaml \
                     --dataset_dir ./yolo_dataset \
                     --dataset_type yolo_v2 \
                     --choice val \
                     --result_dir ./result/eval \
                     --gpu_id 0 \
                     --metrics coco,yolo
