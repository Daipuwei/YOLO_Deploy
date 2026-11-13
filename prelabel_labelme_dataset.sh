#!/bin/bash
# labelme 数据集预标注入口（按需修改 --cfg/--dataset_dir，模型路径经 -o 覆盖或写进 yaml）
# 三种模式示例：

# 1) 覆盖模式（缺省）：预测结果全覆盖写入 json
python tools/prelabel_labelme_dataset.py --cfg ./config/detection.yaml \
                     --dataset_dir ./labelme_dataset \
                     --num_threads 4

# 2) 合并模式：保留已有 json 中与预测框重叠达阈值的标注，追加不重叠的预测框
python tools/prelabel_labelme_dataset.py --cfg ./config/detection.yaml \
                     --dataset_dir ./labelme_dataset \
                     --num_threads 4 \
                     --merge_existing_json --merge_iou_threshold 0.7

# 3) 断点续跑：跳过图像旁已存在同名 json 的图像，重复运行只补未处理的图像
python tools/prelabel_labelme_dataset.py --cfg ./config/detection.yaml \
                     --dataset_dir ./labelme_dataset \
                     --num_threads 4 \
                     --skip_existing_json
