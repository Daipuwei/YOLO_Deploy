# YOLO_Deploy
这是YOLO系列算法在不同部署框架下的Python部署项目。目前模型支持YOLOv5，部署框架支持`ONNXRuntime`、`TensorRT`。

# 版本更新日志
- [2023-04-02]对ONNXRuntime和TensorRT抽象推理引擎模块进行修正，修复不能支持多输入和多输出的bug，并精简抽象检测类接口，版本为`v1.0`;
- [2023-03-29]首次提交代码，支持ONNXRuntme和TensorRT两种部署框架，实现YOLOv5的部署，支持检测图像（集）和视频（集），版本为`v1.0`;

---
# 环境配置
根据不同部署框架选择不同的环境配置文件进行环境配置。
```bash
# onnx
pip install requirement/requirements_onnx.txt

# tensorRT7
pip install requirement/requirements_tensorrt7.txt

# tensorRT8
pip install requirement/requirements_tensorrt8.txt
```

---

# TODO
模型方面：
- [x] YOLOv4
- [x] YOLOv6
- [x] YOLOv7
- [x] YOLOv8
- [x] YOLOX
- [x] PPYOLO
- [x] PPYOLOv2
- [x] ...

部署框架：
- [x] OpenVINO
- [x] Paddle Lite
- [x] TensorFLow lite
- [x] MACE
- [x] MNN
- [x] NCNN
- [x] ...

---
# 推理
```bash
# 图片(集)
python detect.py --cfg ./config/detection.yaml --source abspath/image_dir/ --result_dir ./result
python detect.py --cfg ./config/detection.yaml --source abspath/image --result_dir ./result
# 隔秒检测视频（集）
python detect.py --cfg ./config/detection.yaml --source abspath/video_dir/ --result_dir ./result --interval 1
python detect.py --cfg ./config/detection.yaml --source abspath/video/ --result_dir ./result --interval 1
# 逐帧检测视频（集）,interva=-1代表逐帧检测视频
python detect.py --cfg ./config/detection.yaml --source abspath/video_dir/ --result_dir ./result --interval -1
python detect.py --cfg ./config/detection.yaml --source abspath/video/ --result_dir ./result --interval -1
# 图片或者图片文件夹或者视频或者视频文件夹或者任意组合
python detect.py --cfg ./config/detection.yaml --source abspath/image_and_video/ --result_dir ./result --interval -1
# 视频转VOC数据集,并进行预标注
python video2voc_dataset.py --cfg ./config/detection.yaml --video ./video/ --result_dir ./result --interval 1 --num_threads 4 --confidence_threshold 0.1
# 图像集转VOC数据集,并进行预标注
python imageset2voc_dataset.py --cfg ./config/detection.yaml --imageset ./image/ --result_dir ./result --num_threads 4 --confidence_threshold 0.1
```
