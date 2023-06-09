#! /bin/bash
# 图片
python detect.py --cfg ./config/detection.yaml --source ./image/ --result_dir ./result
# 视频,逐帧检测
python detect.py --cfg ./config/detection.yaml --source ./video/ --result_dir ./result --interval -1
# 视频,隔秒检测
python detect.py --cfg ./config/detection.yaml --source ./video/ --result_dir ./result --interval 1
# 图片或者图片文件夹或者视频或者视频文件夹或者任意组合
python detect.py --cfg ./config/detection.yaml --source ./video/ --result_dir ./result --interval -1
# 视频转VOC数据集,并进行预标注
python video2voc_dataset.py --cfg ./config/detection.yaml --video ./video/ --result_dir ./result --interval 1 --num_threads 4 --confidence_threshold 0.1
# 图像集转VOC数据集,并进行预标注
python imageset2voc_dataset.py --cfg ./config/detection.yaml --imageset ./image/ --result_dir ./result --num_threads 4 --confidence_threshold 0.1
# VOC数据集转COCO数据集
python voc2coco.py --voc_dataset_dir ../VOC2007 --coco_dataset_dir ../VOC2007-COCO --class_name_path ./model_data/voc_names.txt --choices "train" "val" --save_image
# COCO数据集评测
python test.py --cfg ./config/detection.yaml --dataset_dir ../COC2017 --result_dir ./result/yolov5s/COC2017 --choice 'val'  --dataset_type 'coco' --iou_threshold 0.5 --confidence_threshold 0.001 --save_image
# VOC数据集评测
python test.py --cfg ./config/detection.yaml --dataset_dir ../VOC2007 --result_dir ./result/yolov5s/COCO217 --choice 'val'  --dataset_type 'voc' --iou_threshold 0.5 --confidence_threshold 0.001 --save_image
