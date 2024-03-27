# YOLOv8
这是YOLOv8的多框架Python部署说明,目前已支持`ONNXRuntime`、`TensorRT`等框架。

# 1. 模型性能与速度对比
模型性能和速度评估所用硬件与软件配置如下所示。

|     名称      |                 型号(版本)                  |
|:-----------:|:---------------------------------------:|
|     CPU     | Intel(R) Core(TM) i7-8700 CPU @ 3.20GHz |
|     GPU     |             **RTX2080 ti**              |
|    CUDA     |                  11.6                   |
| ONNXRuntime |                 1.10.0                  |
|  TensorRT   |                 8.6.1.6                 |
|   Python    |                   3.8                   |


## 1.1 ONNX
COCO2017数据集上的性能如下,**batchsize=1,confidence_threshold=0.001,iou_threshold=0.5**。

|     模型名称     |   分辨率   | mAP@0.5 | mAP@0.5：0.95 | 预处理时间(ms) | 前向推理时间(ms) | 后处理时间(ms) | 推理时间(ms) |
|:------------:|:-------:|:-------:|:------------:|:---------:|:----------:|:---------:|:--------:|
|   yolov8x    | 640x640 |  55.86  |    42.07     |  3.1296   |  28.1961   |  1.5722   | 32.8979  |
|   yolov8l    | 640x640 |  55.39  |    41.55     |  2.8313   |  18.7267   |  1.5355   | 23.0934  |
|   yolov8m    | 640x640 |  53.73  |    39.84     |  2.8918   |  12.8224   |  1.6447   | 17.3590  |
|   yolov8s    | 640x640 |  49.91  |    36.14     |  2.9600   |   8.4771   |  1.6749   | 13.1119  |
|   yolov8n    | 640x640 |  43.47  |    30.52     |  3.5648   |   6.3066   |  1.8407   | 11.7122  |

## 1.2 TensorRT
COCO2017数据集上的性能如下,**batchsize=1,confidence_threshold=0.001,iou_threshold=0.5**，用于生成TensorRT INT8模型的校准集为：[coco_calib](https://drive.google.com/drive/folders/1s7jE9DtOngZMzJC1uL307J2MiaGwdRSI?usp=sharing)。

|  模型名称   |   分辨率   | mode | mAP@0.5 | mAP@0.5：0.95 | 预处理时间(ms) | 前向推理时间(ms) | 后处理时间(ms) | 推理时间(ms) |
|:-------:|:-------:|:----:|:-------:|:------------:|:---------:|:----------:|:---------:|:--------:|
| yolov8x | 640x640 | fp32 |  55.86  |    42.07     |  2.6950   |  20.2064   |  1.7686   | 24.6700  |
| yolov8x | 640x640 | fp16 |  55.89  |    42.09     |  3.0392   |   9.2403   |  1.7634   | 14.0428  |
| yolov8x | 640x640 | int8 |  54.05  |    40.21     |  2.8290   |   6.3046   |  1.7592   | 10.8928  |
|         |         |      |         |              |           |            |           |          |
| yolov8l | 640x640 | fp32 |  55.39  |    41.55     |  2.9625   |  12.9534   |  1.8149   | 17.7308  |
| yolov8l | 640x640 | fp16 |  55.40  |    41.54     |  2.7436   |   6.6781   |  1.7631   | 11.1848  |
| yolov8l | 640x640 | int8 |  53.35  |    39.46     |  2.9937   |   4.7986   |  1.7765   |  9.5688  |
|         |         |      |         |              |           |            |           |          |
| yolov8m | 640x640 | fp32 |  53.73  |    39.84     |  2.8549   |   9.4062   |  1.8530   | 14.1140  |
| yolov8m | 640x640 | fp16 |  53.74  |    39.83     |  2.6309   |   4.9532   |  1.8145   |  9.3986  |
| yolov8m | 640x640 | int8 |  50.21  |    36.38     |  2.6557   |   3.7979   |  1.8109   |  8.2645  |
|         |         |      |         |              |           |            |           |          |
| yolov8s | 640x640 | fp32 |  49.91  |    36.14     |  2.7904   |   5.4312   |  1.8350   | 10.0567  |
| yolov8s | 640x640 | fp16 |  49.92  |    36.14     |  2.7613   |   2.9235   |  1.8599   |  7.5447  |
| yolov8s | 640x640 | int8 |  47.71  |    34.39     |  2.7596   |   2.5479   |  1.8426   |  7.1501  |
|         |         |      |         |              |           |            |           |          |
| yolov8n | 640x640 | fp32 |  43.47  |    30.52     |  2.8630   |   3.3231   |  1.9045   |  8.0906  |
| yolov8n | 640x640 | fp16 |  43.49  |    30.51     |  2.8988   |   2.2859   |  1.8847   |  7.0693  |
| yolov8n | 640x640 | int8 |  36.09  |    25.10     |  2.9873   |   2.1859   |  1.8528   |  7.0259  |


---
# 2. 模型转换
## 2.1 Pytorch2ONNX
首先下载[ultralytics/yolov8](https://github.com/ultralytics/ultralytics)。
```bash
git clone https://github.com/ultralytics/ultralytics
```
修改`ultralytics/nn/modules/head.py`中的`Detect`类的`forward`函数先关代码:
```python
# 修改前
y = torch.cat((dbox, cls.sigmoid()), 1)

# 修改后
y = torch.cat((dbox.transpose(1,2), cls.sigmoid().transpose(1,2)), -1)
```
然后将本项目中的`tools/yolov8_pytorch2onnx.py`脚本复制到YOLOv8源代码中，并重命名为`pytorch2onnx.py`，然后在官网下载对应模型用于生成yolov8的onnx模型。其中命令行参数含义如下：`model_path`代表模型文件路径，`imgsz`为模型输入图像尺度，`simplify`代表是否使用onnx-simplify对onnx模型进行简化，`half`代表是否将pytorch模型转换为fp16格式，`opset`代表onnx的算子版本，默认为11。
```bash
python pytorch2onnx.py --model_path yolov8n.pt \
       --opset 11 \
       --simplify \
       --imgsz 640 640 \
       --half
```

## 2.2 ONNX2TensorRT
TensorRT模型分成fp32、fp16和int8三种模型，以yolov8n为例，转换命令如下。其中命令行参数含义如下：`onnx_model_path`为ONNX模型路径，`tensorrt_model_path`为TensorRT模型路径，`input_shape`为onnx模型输入尺度，`mode`代表TensorRT模型模式，只能选择fp32、fp16和int8三者之一，`model_type`代表模型类型，用于初始化指定模型的TensorRT模型int8校准类，`calibrator_image_dir`代表TensorRT模型int8校准集路径，`calibrator_table_path`代表TensorRT模型int8校准表路径，`data_type`代表int8校准类当中数据张量类型，默认为float32。
```bash
# fp32
python onnx2tensorrt.py --onnx_model_path ./model_data/yolov8n.onnx \
       --tensorrt_model_path ./model_data/yolov8n.trt \
       --input_shape 1 3 640 640 \
       --mode fp32

# fp16
python onnx2tensorrt.py --onnx_model_path ./model_data/yolov8n.onnx \
       --tensorrt_model_path ./model_data/yolov8n.trt \
       --input_shape 1 3 640 640 \
       --mode fp16

# int8
python onnx2tensorrt.py --onnx_model_path ./model_data/yolov8n.onnx \
       --tensorrt_model_path ./model_data/yolov8n.trt \
       --input_shape 1 3 640 640 \
       --mode int8 \
       --model_type yolov8 \
       --calibrator_image_dir ./image/coco_calib/ \
       --calibrator_table_path ./model_data/yolov8n_coco_calibrator_table.cache \
       --data_type float32
```

---
# 3. 功能介绍
## 3.1 检测图像和视频
`detect.py`主要功能为检测图像(集)和视频(集)。其中命令行参数含义如下：`cfg`为模型参数配置文件路径，`source`为待检测的图像(集)或视频(集)路径，`result_dir`为检测结果文件夹路径，`interval`为视频抽帧频率，若为-1代表逐帧检测，`num_threads`为线程数，`print_detection_result`代表是否打印检测结果，`-o`选项自定义相关参数来对yaml配置文件中的参数进行更新。
```bash
# 检测图片(集)
python detect.py --cfg ./config/yolov8.yaml \
       --source ./image/coco_calib/ \
       --result_dir ./result/image_video/ \
       -o DetectionModel.engine_model_path=./model_data/yolov8s.trt.fp16 DetectionModel.confidence_threshold=0.5

# 检测视频(集)， 逐帧检测
python detect.py --cfg ./config/yolov8.yaml \
       --source ./video/1.dav \
       --result_dir ./result/image_video/ \
       --interval -1 \
       -o DetectionModel.model_type=yolov8 DetectionModel.engine_model_path=./model_data/yolov8s.trt.fp16 DetectionModel.confidence_threshold=0.1

# 检测视频(集)，隔秒检测
python detect.py --cfg ./config/yolov8.yaml \
       --source ./video/1.dav \
       --result_dir ./result/image_video/ \
       --interval 1 \
       -o DetectionModel.model_type=yolov8 DetectionModel.engine_model_path=./model_data/yolov8s.trt.fp16 DetectionModel.confidence_threshold=0.1

# 图片或者图片文件夹或者视频或者视频文件夹或者任意组合
python detect.py --cfg ./config/yolov8.yaml \
       --source ./image_video/ \
       --result_dir ./result/image_video/ \
       --interval -1 \
       -o DetectionModel.model_type=yolov8 DetectionModel.engine_model_path=./model_data/yolov8s.trt.fp16 DetectionModel.confidence_threshold=0.1
```

## 3.2 评估模型性能
`test.py`主要功能为评估模型在VOC格式数据集或者COCO格式数据集上的性能。其中命令行参数含义如下：`cfg`为模型参数配置文件路径，`dataset_dir`为数据集路径，`dataset_type`为数据集类型，候选值为voc和coco，`choice`为数据集子集类型,`result_dir`为评估结果文件夹路径，`save_image`代表是否保存检测图像，`export_time`代表是否输出推理时间选项，`print_detection_result`代表是否打印检测结果，`-o`选项自定义相关参数来对yaml配置文件中的参数进行更新。
```bash
# COCO格式数据集
python test.py --cfg ./config/yolov8.yaml \
       --dataset_dir /home/dpw/deeplearning/coco2017 \
       --dataset_type coco \
       --choice val \
       --result_dir ./result/test \
       --save_image \
       --export_time \
       --print_detection_result \
       -o DetectionModel.engine_type=onnx DetectionModel.engine_model_path=./model_data/yolov8n.onnx DetectionModel.confidence_threshold=0.001 DetectionModel.iou_threshold=0.5

# VOC格式数据集
python test.py --cfg ./config/yolov8.yaml \
       --dataset_dir /home/dpw/deeplearning/coco2017_voc \
       --dataset_type voc \
       --choice val \
       --result_dir ./result/test \
       --save_image \
       --export_time \
       --print_detection_result \
       -o DetectionModel.engine_type=onnx DetectionModel.engine_model_path=./model_data/yolov8n.onnx DetectionModel.confidence_threshold=0.001 DetectionModel.iou_threshold=0.5
```

## 3.3 图像视频预标注成VOC数据集
`imageset2voc_dataset.py`主要功能是对图像集进行预标注生成VOC数据集，`video2voc_dataset.py`主要功能是对视频(集)进行预标注生成VOC数据集。其中命令行参数含义如下：`cfg`为模型参数配置文件路径，`imageset`为图像集文件夹路径，`result_dir`为评估结果文件夹路径，`num_threads`为线程数，`interval`为视频抽帧频率，若为-1代表逐帧检测，`print_detection_result`代表是否打印检测结果，`-o`选项自定义相关参数来对yaml配置文件中的参数进行更新。
```bash
# 视频转VOC数据集,并进行预标注
python video2voc_dataset.py --cfg ./config/yolov8.yaml \
       --video ./video/ \
       --result_dir ./result_dir/voc_dataset \
       --num_threads 4 \
       --interval -1 \
       --print_detection_result \
       -o DetectionModel.model_type=yolov8 DetectionModel.engine_model_path=./model_data/yolov8s.trt.fp16 DetectionModel.confidence_threshold=0.5 DetectionModel.iou_threshold=0.5

# 图像集转VOC数据集,并进行预标注
python imageset2voc_dataset.py --cfg ./config/yolov8.yaml \
       --imageset ./image/coco_calib/ \
       --result_dir ./result_dir/voc_dataset \
       --num_threads 4 \
       -o DetectionModel.model_type=yolov8 DetectionModel.engine_model_path=./model_data/yolov8s.trt.fp16 DetectionModel.confidence_threshold=0.5 DetectionModel.iou_threshold=0.5
```

## 3.4 图像视频预标注成Labelme数据集
`imageset2labelme_dataset.py`主要功能是对图像集进行预标注生成Labelme数据集，`video2labelme_dataset.py`主要功能是对视频(集)进行预标注生成Labelme数据集。其中命令行参数含义如下：`cfg`为模型参数配置文件路径，`imageset`为图像集文件夹路径，`result_dir`为评估结果文件夹路径，`num_threads`为线程数，`interval`为视频抽帧频率，若为-1代表逐帧检测，`print_detection_result`代表是否打印检测结果，`-o`选项自定义相关参数来对yaml配置文件中的参数进行更新。
```bash
# 视频转VOC数据集,并进行预标注
python video2labelme_dataset.py --cfg ./config/yolov8.yaml \
       --video ./video/ \
       --result_dir ./result_dir/labelme_dataset \
       --num_threads 4 \
       --interval -1 \
       --print_detection_result \
       -o DetectionModel.model_type=yolov8 DetectionModel.engine_model_path=./model_data/yolov8s.trt.fp16 DetectionModel.confidence_threshold=0.5 DetectionModel.iou_threshold=0.5

# 图像集转VOC数据集,并进行预标注
python imageset2labelme_dataset.py --cfg ./config/yolov8.yaml \
       --imageset ./image/coco_calib/ \
       --result_dir ./result_dir/labelme_dataset \
       --num_threads 4 \
       -o DetectionModel.model_type=yolov8 DetectionModel.engine_model_path=./model_data/yolov8s.trt.fp16 DetectionModel.confidence_threshold=0.5 DetectionModel.iou_threshold=0.5
```