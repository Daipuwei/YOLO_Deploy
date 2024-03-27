# YOLO_Deploy
这是YOLO系列算法在不同部署框架下的Python部署项目。目前模型支持YOLOv5、YOLOv8和YOLOS，部署框架支持`ONNXRuntime`、`TensorRT`。

# 版本更新日志
- **[2024-03-27]**新增yolov8和yolos模型实现，并对相关代码结构和bug进行进一步优化，版本为`v1.0`；
- **[2024-01-18]**升级代码，支持TensorRT INT8后量化，TensorRT模型生成脚本独立，优化TensorRT和ONNXRuntime推理引擎，并编写重构视频和图像集预标注脚本并支持VOC和Labelme两种格式，版本为`v1.0`；
- **[2023-10-02]**升级代码，命令行解析类支持自定义更新任意多个yaml文件配置参数，并对多个脚本的代码结构进行改造，版本为`v1.0`;
- **[2023-04-16]**完成模型性能测试脚本，性能指标支持mAP@0.5、mAP@0.5:0.95、mAR@0.5：0.95、每类目标的AP@0.5和AR@0.5，支持VOC数据集和COCO数据集，版本为`v1.0`;
- **[2023-04-02]**对ONNXRuntime和TensorRT抽象推理引擎模块进行修正，修复不能支持多输入和多输出的bug，并精简抽象检测类接口，版本为`v1.0`;
- **[2023-03-29]**首次提交代码，支持ONNXRuntme和TensorRT两种部署框架，实现YOLOv5的部署，支持检测图像（集）和视频（集），版本为`v1.0`;

---
# 教程
- 相关Python环境安装详见[INSTALL](docs/INSTALL.md)；
- yolov5的部署教程详见[yolov5_tutorial](docs/models/yolov5.md);
- yolov8的部署教程详见[yolov8_tutorial](docs/models/yolov8.md);
- yolos的部署教程详见[yolos_tutorial](docs/models/yolos.md);
