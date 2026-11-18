# DetectionTrackToolkit-py
这是多检测模型与多跟踪算法工具箱，支持多推理框架Python推理。

检测模型支持：

- [x] AnchorBase:
  - [x] YOLOv5
  - [x] YOLOv7
- [x] AnchorFree:
  - [x] YOLOX
  - [x] DFLYOLO: YOLOv6、YOLOv8、YOLOv9、YOLOv10、YOLOv11、YOLOv12、YOLOv13、YOLO26

推理框架支持：
- [x] ONNXRuntime
- [x] TensorRT


功能支持：
- [x] 检测图片、图片集、视频集及其任意组合（不包含跟踪），支持多实例版本多线程；
- [x] 对VOC、COCO、YOLO格式数据集评测多推理框架下检测模型的检测性能，检测性能指标口径包含COCO API和Ultralytics API两种
- [x] Labelme格式数据集与标注，支持覆盖/合并/断点续跑三种预标注模式与多目录批量遍历（`prelabel_traverse_subdirs.sh`），支持多实例版本多线程；

---

# TODO


- [x] yolox、dflyolo的单输出onnx模型输出前是否带transpose节点时后处理脚本需要新增支持
- [x] yolov5、yolox、dflyolo的多输出onnx模型在后处理阶段需要新增对应解码功能
- [x] TensorRT推理引擎代码进一步优化，重点适配TensorRT7以下、TensorRT8、TensorRT10不同版本API差异
- [x] ONNXRuntime推理引擎老版本接口适配
- [ ] OpenVINO推理引擎集成
- [ ] ONNXRuntime、TensorR、OpenVINO异步推理实现
- [ ] ByteTrack为代表的跟踪算法集成及其对应的性能测试与检测脚本实现


--- 


# 版本更新日志
- **[2026-11-18]**:修复voc和yolo数据集加载中图像名存在.导致读取图像失败的bug，版本为`v3.0`；
- **[2026-11-13]**:项目整体更名为**DetectionTrackToolkit-py**，并对代码做了彻底性架构重构：DetectionModel作为编排层装配推理引擎/预处理/后处理三块（模型、引擎、预处理、后处理四大注册器驱动），实现预处理/推理/后处理三阶段解耦，完成不同YOLO模型之间可复用；并修复推理、评估、数据集加载、指标计算、预处理/后处理中的多处重大bug，之前版本代码完全弃用，版本为`v3.0`；
- **[2025-04-08]**:修复docker镜像部署脚本bug，版本为`v2.0`；
- **[2025-03-21]**:新增yolov4、yolov6、yolov7、yolov9、yolov10、yolo11、yolov12、yolox和ppyoloe模型实现，新增x86部署docker镜像说明，并将TensorrRT模型转换独立成单独项目，版本为`v2.0`；
- **[2024-04-14]**:新增增加模型、推理引擎、数据集加载器和TensorRT的INT8量化校准集注册机制，进一步优化代码结构，版本为`v1.0`；
- **[2024-03-27]**:新增yolov8和yolos模型实现，并对相关代码结构和bug进行进一步优化，版本为`v1.0`；
- **[2024-01-18]**:升级代码，支持TensorRT INT8后量化，TensorRT模型生成脚本独立，优化TensorRT和ONNXRuntime推理引擎，并编写重构视频和图像集预标注脚本并支持VOC和Labelme两种格式，版本为`v1.0`；
- **[2023-10-02]**:升级代码，命令行解析类支持自定义更新任意多个yaml文件配置参数，并对多个脚本的代码结构进行改造，版本为`v1.0`;
- **[2023-04-16]**:完成模型性能测试脚本，性能指标支持mAP@0.5、mAP@0.5:0.95、mAR@0.5：0.95、每类目标的AP@0.5和AR@0.5，支持VOC数据集和COCO数据集，版本为`v1.0`;
- **[2023-04-02]**:对ONNXRuntime和TensorRT抽象推理引擎模块进行修正，修复不能支持多输入和多输出的bug，并精简抽象检测类接口，版本为`v1.0`;
- **[2023-03-29]**:首次提交代码，支持ONNXRuntme和TensorRT两种部署框架，实现YOLOv5的部署，支持检测图像（集）和视频（集），版本为`v1.0`;

# 教程与文档
- 快速上手（环境安装→配置说明→检测/预标注/评估三大工具）：见[QuickStart](docs/QuickStart.md)；
- 环境安装（裸机从零）：见[INSTALL](docs/INSTALL.md)；Docker容器部署：见[docker/README.md](docker/README.md)；
- 评测指标口径详解（YOLO与COCO API两种口径差异）：见[METRICS](docs/METRICS.md)；
- 项目架构与扩展指南（三块装配/注册器/即插即用）：见[ARCHITECTURE](docs/ARCHITECTURE.md)；
- TenosrRT模型转换教程详见[TensorRT-Tools-py](https://github.com/Daipuwei/TensorRT-Tools-py);
