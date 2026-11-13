#!/bin/bash
# 图像/视频检测入口（按需修改 --cfg/--source 与模型参数；--source 支持单文件或目录，
# 目录会递归遍历子目录并按扩展名自动分流图像/视频）

# 1) 单张图片
python tools/detect.py --cfg ./config/detection.yaml \
                       --source ./test.jpg \
                       --result_dir ./result/detect \
                       --gpu_id 0

# 2) 单个视频
python tools/detect.py --cfg ./config/detection.yaml \
                       --source ./test.mp4 \
                       --result_dir ./result/detect \
                       --gpu_id 0

# 3) 图片文件夹（递归遍历子目录）
python tools/detect.py --cfg ./config/detection.yaml \
                       --source ./images \
                       --result_dir ./result/detect \
                       --gpu_id 0

# 4) 视频文件夹（递归遍历子目录）
python tools/detect.py --cfg ./config/detection.yaml \
                       --source ./videos \
                       --result_dir ./result/detect \
                       --gpu_id 0

# 5) 图片视频混合文件夹（递归遍历子目录，自动分流图像与视频）
python tools/detect.py --cfg ./config/detection.yaml \
                       --source ./media \
                       --result_dir ./result/detect \
                       --gpu_id 0
