# YOLO_Deploy
这是YOLO系列算法在不同部署框架下的Python部署项目。目前模型支持YOLOv5，部署框架支持`ONNXRuntime`、`TensorRT`。

# 版本更新日志
- [2023-10-02]升级代码，命令行解析类支持自定义更新任意多个yaml文件配置参数，并对多个脚本的代码结构进行改造，版本为`v1.0`;
- [2023-04-16]完成模型性能测试脚本，性能指标支持mAP@0.5、mAP@0.5:0.95、mAR@0.5：0.95、每类目标的AP@0.5和AR@0.5，支持VOC数据集和COCO数据集，版本为`v1.0`;
- [2023-04-02]对ONNXRuntime和TensorRT抽象推理引擎模块进行修正，修复不能支持多输入和多输出的bug，并精简抽象检测类接口，版本为`v1.0`;
- [2023-03-29]首次提交代码，支持ONNXRuntme和TensorRT两种部署框架，实现YOLOv5的部署，支持检测图像（集）和视频（集），版本为`v1.0`;

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
# 部署
该项目中的模型部署目前支持检测图片(集)和视频(集)，VOC和COCO格式数据集的模型性能评测,具体操作文档请相见[quickstart](./doc/quickstart.md)
