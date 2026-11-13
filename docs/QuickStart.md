# QuickStart
这是DetectionTrackToolkit-py的快速开始说明，覆盖环境安装、配置文件说明、
图像视频检测、预标注与模型评估。评测指标语义详见[METRICS.md](./METRICS.md)。

---
# 1. 环境安装
环境安装根据部署形态二选一，均按文档流程执行即可，此处不再重复：

| 部署形态 | 文档 | 说明 |
|:---:|:---:|:---:|
| Docker容器（推荐） | [docker/README.md](../docker/README.md) | 拉取英伟达官方CUDA镜像 → init_docker.sh启动容器 → SSH连接 → 容器内安装 |
| 裸机 | [INSTALL.md](./INSTALL.md) | 创建虚拟环境 → 按需安装（默认CUDA模式）→ 准备模型配置 → 验证 |

安装完成后可用以下命令验证：

```bash
detect --help
eval_dataset --help        # 注意：评测命令名为eval_dataset（eval与bash内建冲突）
prelabel --help
```

---
# 2. 配置文件（yaml）说明
## 2.1 配置结构
所有工具均通过`--cfg`指定yaml配置文件（主模板：`config/detection.yaml`，
完整注释见该文件），结构如下：

| 配置键 | 必填 | 缺省值 | 说明 |
|:---:|:---:|:---:|:---|
| DetectionModel.model_type | 是 | - | 模型类型：yolov5/yolov7/yolox/yolov6/v8/v9/v10/v11/v12/v13/yolo26/ppyoloe |
| DetectionModel.engine_type | 是 | - | 推理引擎：onnxruntime（CUDA自动优先）或tensorrt |
| DetectionModel.engine_model_path | 是 | - | 模型文件路径（.onnx或.trt/.engine） |
| DetectionModel.class_name_path | 是 | - | 类别名文件路径（每行一个类别名） |
| DetectionModel.confidence_threshold | 是 | - | 置信度阈值，低于该值的框被过滤 |
| DetectionModel.iou_threshold | 是 | - | NMS抑制阈值（与评测AP的IoU阈值无耦合，见METRICS.md） |
| DetectionModel.is_normalize | 否 | True | 是否÷255归一化（yolov5/DFL读取；div255已融入ONNX时设False） |
| DetectionModel.resize_mode | 否 | keep_aspect_ratio | 缩放方式：keep_aspect_ratio等比缩放+letterbox填充 / resize直接拉伸（拉伸时坐标按h/w双轴独立精确还原） |
| DetectionModel.pad_mode | 否 | top_left | letterbox填充锚定：top_left填充全在右下（还原无偏移）/ center居中填充（ultralytics口径，还原减填充偏移；两种模式检测精度等价） |
| DetectionModel.bbox_format | 否 | xyxy | bbox坐标布局：xyxy/cxcywh（yolox/DFL读取） |
| DetectionModel.output_layout | 否 | grid_first | 输出形状布局：grid_first=bs×grid_num×output_dim；dim_first=bs×output_dim×grid_num（导出带transpose时，yolox/DFL读取） |
| DetectionModel.nms_mode | 否 | class_agnostic | NMS模式：class_agnostic整体过滤/per_class两阶段按类过滤 |
| DetectionModel.class_agnostic_threshold | 否 | None | per_class模式第二阶段全局过滤阈值（缺省None=用iou_threshold过滤，必须执行；0=禁用；>0=自定义阈值） |
| Log.* | 否 | - | 日志配置：log_path/轮转大小/保留份数/各级别/enqueue，见detection.yaml注释 |

引擎调优参数（gpu_id/intra_op_threads/cudnn_conv_algo_search等）不出现在
yaml中，缺省值与语义见`engine/onnxruntime/onnxruntime_engine.py`常量区。

## 2.2 命令行覆盖（-o）
命令行`-o key=value`（点分键`DetectionModel.key=value`）可覆盖yaml中的
任意配置，优先级：命令行 -o > yaml > 代码缺省。示例：

```bash
-o DetectionModel.confidence_threshold=0.001 DetectionModel.iou_threshold=0.5 \
   DetectionModel.engine_model_path=./model_data/xxx.onnx
```

## 2.3 模型配置模板
`config/detection.yaml`为主配置模板（含全部参数注释说明），
`config/detection/`下提供12个模型模板（yolov5~13/yolox/ppyoloe，均带注释），
`config/depth/`为深度图检测的工作配置。新模型按需从模板复制修改。

---
# 3. 检测图像和视频
`detect.py`主要功能为检测图像和视频。其中命令行参数含义如下：
`cfg`为模型参数配置文件路径，`source`为待检测的路径，支持五种形态——
单张图片、单个视频、图片文件夹、视频文件夹、图片视频混合文件夹
（目录会递归遍历子目录并按扩展名自动分流图像/视频），
`result_dir`为检测结果文件夹路径（结果按
模型名/时间戳归档），`interval`为视频抽帧频率（-1为逐帧检测），
`num_threads`为线程数（每线程一个独立模型实例），`gpu_id`为GPU设备号，
`print_detection_result`代表是否打印检测结果，`export_time`代表是否输出
检测耗时，`-o`选项自定义相关参数对yaml配置进行更新。

命令样例与仓库根目录`detect.sh`一致（按需复制其中命令块）：

```bash
# 单张图片
python tools/detect.py --cfg ./config/detection.yaml \
       --source ./test.jpg \
       --result_dir ./result/detect \
       --gpu_id 0

# 单个视频
python tools/detect.py --cfg ./config/detection.yaml \
       --source ./test.mp4 \
       --result_dir ./result/detect \
       --gpu_id 0

# 图片文件夹（递归遍历子目录）
python tools/detect.py --cfg ./config/detection.yaml \
       --source ./images \
       --result_dir ./result/detect \
       --gpu_id 0

# 视频文件夹（递归遍历子目录）
python tools/detect.py --cfg ./config/detection.yaml \
       --source ./videos \
       --result_dir ./result/detect \
       --gpu_id 0

# 图片视频混合文件夹（自动分流图像与视频）
python tools/detect.py --cfg ./config/detection.yaml \
       --source ./media \
       --result_dir ./result/detect \
       --gpu_id 0
```

---
# 4. 图像预标注（Labelme数据集）
`prelabel_labelme_dataset.py`主要功能是使用检测模型对labelme数据集进行
预标注，生成labelme标注json供人工复核。其中命令行参数含义如下：
`cfg`为模型参数配置文件路径，`dataset_dir`为labelme数据集路径（根目录
含images子目录或一级子目录为数据集），`num_threads`为线程数，
`print_detection_result`代表是否打印检测结果，`export_time`代表是否输出
检测耗时，`merge_existing_json`为合并模式（保留已有json标注中与预测框
重叠度达`merge_iou_threshold`的框，追加不重叠的预测框；缺省False为
覆盖模式），`skip_existing_json`为断点续跑（跳过图像旁已存在同名json的
图像，json与图像同目录，重复运行只补未处理的图像），
`-o`选项自定义相关参数对yaml配置进行更新。

**多目录批量预标注**：仓库根目录`run.sh`为便捷入口，链路
`run.sh → tools/prelabel_traverse_subdirs.sh → prelabel_labelme_dataset.py`。
通用遍历脚本对父目录下每个一级子目录逐个预标注，全部配置
（`--cfg/--model/--gpu_id/--num_threads/--conf/--iou`及`-o`透传等）
命令行传入，引擎类型按模型扩展名自动判定（.onnx→onnxruntime，
.trt/.engine→tensorrt），支持`--skip_existing_json`断点续跑，
单个目录失败不中断、结束统一汇总失败清单。示例：

```bash
bash tools/prelabel_traverse_subdirs.sh \
    /path/to/unlabel \
    --cfg ./config/detection.yaml \
    --model /path/to/model.onnx \
    --gpu_id 0 --num_threads 4 \
    --skip_existing_json
```

命令样例与仓库根目录`prelabel_labelme_dataset.sh`一致（按需复制其中命令块）：

```bash
# 覆盖模式预标注（预测结果全覆盖）
python tools/prelabel_labelme_dataset.py --cfg ./config/detection.yaml \
       --dataset_dir ./labelme_dataset \
       --num_threads 4

# 合并模式预标注（保留已有标注，追加不重叠预测框）
python tools/prelabel_labelme_dataset.py --cfg ./config/detection.yaml \
       --dataset_dir ./labelme_dataset \
       --num_threads 4 \
       --merge_existing_json --merge_iou_threshold 0.7

# 断点续跑（跳过已生成 json 的图像，重复运行只补未处理）
python tools/prelabel_labelme_dataset.py --cfg ./config/detection.yaml \
       --dataset_dir ./labelme_dataset \
       --num_threads 4 \
       --skip_existing_json
```

---
# 5. 评估模型性能
`eval.py`主要功能为评估模型在VOC/COCO/YOLO格式数据集上的性能，同时输出
COCO API与Ultralytics YOLO两种口径的指标（口径差异见[METRICS.md](./METRICS.md)）。
其中命令行参数含义如下：`cfg`为模型参数配置文件路径，`dataset_dir`为
数据集路径，`dataset_type`为数据集类型（候选值voc/coco/yolo_v1/yolo_v2），`choice`
为数据集子集（单个或逗号分隔多个，如val或val,test），`result_dir`为
评估结果文件夹路径，`save_image`代表是否保存检测图像，`export_time`
代表是否输出推理耗时，`print_detection_result`代表是否打印检测结果，
`metrics`为评估指标集合（候选值coco与/或yolo），`-o`选项自定义相关参数
对yaml配置进行更新。

命令样例与仓库根目录`eval.sh`一致（按需复制其中命令块）：

```bash
# VOC 格式数据集（Annotations/JPEGImages/ImageSets）
python tools/eval.py --cfg ./config/detection.yaml \
       --dataset_dir ./VOC2007 \
       --dataset_type voc \
       --choice val \
       --result_dir ./result/eval \
       --gpu_id 0 \
       --metrics coco,yolo

# COCO 格式数据集
python tools/eval.py --cfg ./config/detection.yaml \
       --dataset_dir ./coco_dataset \
       --dataset_type coco \
       --choice val \
       --result_dir ./result/eval \
       --gpu_id 0 \
       --metrics coco,yolo

# YOLO v1 格式数据集（父目录含 images/、labels/ 与子集 txt 清单）
python tools/eval.py --cfg ./config/detection.yaml \
       --dataset_dir ./yolo_dataset \
       --dataset_type yolo_v1 \
       --choice val \
       --result_dir ./result/eval \
       --gpu_id 0 \
       --metrics coco,yolo

# YOLO v2 格式数据集（父目录下每个子集文件夹含 images/、labels/）
python tools/eval.py --cfg ./config/detection.yaml \
       --dataset_dir ./yolo_dataset \
       --dataset_type yolo_v2 \
       --choice val \
       --result_dir ./result/eval \
       --gpu_id 0 \
       --metrics coco,yolo
```

评估结果落盘于`result_dir`下：`result.txt`（指标汇总，含混淆矩阵表格）、
`gt_result.json`/`detection_result.json`（COCO格式，供二次分析）。
