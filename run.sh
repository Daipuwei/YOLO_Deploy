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

# COCO数据集评测
python voc2coco.py --voc_dataset_dir ../WIDERFACE --coco_dataset_dir ../WIDERFACE-COCO --class_name_path ./model_data/face_names.txt --choices "train" "val" --save_image
sudo cp coco/val/json/path ./result/xxx/gt_result.json
python test_coco.py --cfg ./config/detection.yaml --coco_image_dir ../WIDERFACE/val --result_dir ./result/ --model_name 'yolov5s_RFEM_MultiSEAM' #--save_image
python compute_mAP_COCO.py --gt_json_path ./result/WIDERFACE/yolov5s_RFEM_MultiSEAM/gt_result.json --dr_json_path ./result/WIDERFACE/yolov5s_RFEM_MultiSEAM/dr_result.json

# VOC数据集评测
python test_voc.py --cfg ./config/detection.yaml --voc_dataset_dir ../WIDERFACE --result_dir ./mAP/input/WIDERFACE --choice 'val' --model_name 'yolov5s_RFEM_MultiSEAM' #--save_image
python ./mAP/get_gt_txt.py --input_dir ./mAP/input/WIDERFACE/yolov5s_RFEM_MultiSEAM --voc_dataset_dir ../WIDERFACE/ --choice 'val'
python ./mAP/compute_mAP.py --input_dir ./mAP/input/WIDERFACE/yolov5s_RFEM_MultiSEAM
