# YOLOv5
这是YOLOv5的多框架Python部署说明,目前已支持`ONNXRuntime`、`TensorRT`等框架。

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

|     模型名称     |   分辨率   | mAP@0.5 | mAP@0.5：0.95 | 预处理时间(ms) | 前向推理时间(ms) | 后处理时间(ms) | 检测时间(ms) |
|:------------:|:-------:|:-------:|:------------:|:---------:|:----------:|:---------:|:--------:|
|   yolov5x    | 640x640 |  55.55  |    40.75     |  3.1832   |  30.4568   |  1.3228   | 34.9628  |
|   yolov5l    | 640x640 |  54.63  |    39.62     |  2.9192   |  18.2695   |  1.4344   | 22.6213  |
|   yolov5m    | 640x640 |  52.10  |    36.79     |  3.1405   |  12.3984   |  1.5684   | 17.1073  |
|   yolov5s    | 640x640 |  47.26  |    31.32     |  2.8427   |   8.9017   |  1.7433   | 13.4878  |
|   yolov5n    | 640x640 |  38.29  |    23.42     |  2.8943   |   6.9528   |  2.1634   | 12.0105  |

## 1.2 TensorRT
COCO2017数据集上的性能如下,**batchsize=1,confidence_threshold=0.001,iou_threshold=0.5**，用于生成TensorRT INT8模型的校准集为：[coco_calib](https://drive.google.com/drive/folders/1s7jE9DtOngZMzJC1uL307J2MiaGwdRSI?usp=sharing)。

|  模型名称   |   分辨率   | mode | mAP@0.5 | mAP@0.5：0.95 | 预处理时间(ms) | 前向推理时间(ms) | 后处理时间(ms) | 检测时间(ms) |
|:-------:|:-------:|:----:|:-------:|:------------:|:---------:|:----------:|:---------:|:--------:|
| yolov5x | 640x640 | fp32 |  55.55  |    40.75     |  2.7848   |  18.8940   |  1.4896   | 23.1684  |
| yolov5x | 640x640 | fp16 |  55.53  |    40.74     |  2.8076   |   8.9436   |  1.5129   | 13.2641  |
| yolov5x | 640x640 | int8 |  51.55  |    47.81     |  2.7859   |   6.4174   |  1.4517   | 10.6550  |
|         |         |      |         |              |           |            |           |          |
| yolov5l | 640x640 | fp32 |  55.39  |    41.55     |  2.9625   |  12.9534   |  1.8149   | 17.7308  |
| yolov5l | 640x640 | fp16 |  55.40  |    41.54     |  2.7436   |   6.6781   |  1.7631   | 11.1848  |
| yolov5l | 640x640 | int8 |  49.87  |    36.00     |  2.7706   |   4.4844   |  1.5090   |  8.7641  |
|         |         |      |         |              |           |            |           |          |
| yolov5m | 640x640 | fp32 |  52.10  |    36.79     |  2.6381   |   9.1410   |  1.7296   | 13.5088  |
| yolov5m | 640x640 | fp16 |  52.12  |    36.78     |  2.9823   |   4.2507   |  1.7160   |  8.9490  |
| yolov5m | 640x640 | int8 |  45.53  |    31.81     |  2.6995   |   3.6122   |  1.6361   |  7.9478  |
|         |         |      |         |              |           |            |           |          |
| yolov5s | 640x640 | fp32 |  47.26  |    31.32     |  2.8229   |   4.7640   |  1.9125   |  9.4994  |
| yolov5s | 640x640 | fp16 |  47.22  |    31.28     |  2.7565   |   2.8639   |  1.9079   |  7.5283  |
| yolov5s | 640x640 | int8 |  40.82  |    26.42     |  2.8719   |   2.6486   |  1.8959   |  7.414   |
|         |         |      |         |              |           |            |           |          |
| yolov5n | 640x640 | fp32 |  38.29  |    23.42     |  2.9671   |   3.2740   |  2.5005   |  8.7416  |
| yolov5n | 640x640 | fp16 |  38.30  |    23.41     |  2.9866   |   2.4703   |  2.4163   |  7.8731  |
| yolov5n | 640x640 | int8 |  25.71  |    14.22     |  3.0220   |   2.4573   |  2.2612   |  7.7405  |


---
# 2. 模型转换
## 2.1 Pytorch2ONNX
首先下载[ultralytics/yolov5](https://github.com/ultralytics/yolov5)。
```bash
git clone https://github.com/ultralytics/yolov5
```
在官网下载对应模型，然后运行如下命令完成Pytorch到ONNX的模型转换。其中命令行参数含义如下：`weights`代表模型文件路径，`imgsz`为模型输入图像尺度，`device`代表运行pytorch的设备，默认为cpu，`simplify`代表是否使用onnx-simplify对onnx模型进行简化，`half`代表是否将pytorch模型转换为fp16格式，`opset`代表onnx的算子版本，默认为11，`include`代表是转换后的模型的格式，要想转换成onnx模型则制定为onnx。
```bash
python export.py --weights yolov5n.pt  \
       --imgsz 640 640 \
       --batch-size 1  \
       --device cpu \
       --simplify \
       --opset 11 \
       --half \
       --include onnx
```

## 2.2 ONNX2TensorRT
TensorRT模型分成fp32、fp16和int8三种模型，以yolov5n为例，转换命令如下。其中命令行参数含义如下：`onnx_model_path`为ONNX模型路径，`tensorrt_model_path`为TensorRT模型路径，`input_shape`为onnx模型输入尺度，`mode`代表TensorRT模型模式，只能选择fp32、fp16和int8三者之一，`model_type`代表模型类型，用于初始化指定模型的TensorRT模型int8校准类，`calibrator_image_dir`代表TensorRT模型int8校准集路径，`calibrator_table_path`代表TensorRT模型int8校准表路径，`data_type`代表int8校准类当中数据张量类型，默认为float32。
```bash
# fp32
python onnx2tensorrt.py --onnx_model_path ./model_data/yolov5n.onnx \
       --tensorrt_model_path ./model_data/yolov5n.trt \
       --input_shape 1 3 640 640 \
       --mode fp32

# fp16
python onnx2tensorrt.py --onnx_model_path ./model_data/yolov5n.onnx \
       --tensorrt_model_path ./model_data/yolov5n.trt \
       --input_shape 1 3 640 640 \
       --mode fp16

# int8
python onnx2tensorrt.py --onnx_model_path ./model_data/yolov5n.onnx \
       --tensorrt_model_path ./model_data/yolov5n.trt \
       --input_shape 1 3 640 640 \
       --mode int8 \
       --model_type yolov5 \
       --calibrator_image_dir ./image/coco_calib/ \
       --calibrator_table_path ./model_data/yolov5n_coco_calibrator_table.cache \
       --data_type float32
```

---
# 3. 功能介绍
## 3.1 检测图像和视频
`detect.py`主要功能为检测图像(集)和视频(集)。其中命令行参数含义如下：`cfg`为模型参数配置文件路径，`source`为待检测的图像(集)或视频(集)路径，`result_dir`为检测结果文件夹路径，`interval`为视频抽帧频率，若为-1代表逐帧检测，`num_threads`为线程数，`print_detection_result`代表是否打印检测结果，`-o`选项自定义相关参数来对yaml配置文件中的参数进行更新。
```bash
# 检测图片(集)
python detect.py --cfg ./config/yolov5.yaml \
       --source ./image/coco_calib/ \
       --result_dir ./result/image_video/ \
       -o DetectionModel.engine_model_path=./model_data/yolov5s.trt.fp16 DetectionModel.confidence_threshold=0.5

# 检测视频(集)， 逐帧检测
python detect.py --cfg ./config/yolov5.yaml \
       --source ./video/1.dav \
       --result_dir ./result/image_video/ \
       --interval -1 \
       -o DetectionModel.model_type=yolov5 DetectionModel.engine_model_path=./model_data/yolov5s.trt.fp16 DetectionModel.confidence_threshold=0.1

# 检测视频(集)，隔秒检测
python detect.py --cfg ./config/yolov5.yaml \
       --source ./video/1.dav \
       --result_dir ./result/image_video/ \
       --interval 1 \
       -o DetectionModel.model_type=yolov5 DetectionModel.engine_model_path=./model_data/yolov5s.trt.fp16 DetectionModel.confidence_threshold=0.1

# 图片或者图片文件夹或者视频或者视频文件夹或者任意组合
python detect.py --cfg ./config/yolov5.yaml \
       --source ./image_video/ \
       --result_dir ./result/image_video/ \
       --interval -1 \
       -o DetectionModel.model_type=yolov5 DetectionModel.engine_model_path=./model_data/yolov5s.trt.fp16 DetectionModel.confidence_threshold=0.1
```

## 3.2 评估模型性能
`test.py`主要功能为评估模型在VOC格式数据集或者COCO格式数据集上的性能。其中命令行参数含义如下：`cfg`为模型参数配置文件路径，`dataset_dir`为数据集路径，`dataset_type`为数据集类型，候选值为voc和coco，`choice`为数据集子集类型,`result_dir`为评估结果文件夹路径，`save_image`代表是否保存检测图像，`export_time`代表是否输出推理时间选项，`print_detection_result`代表是否打印检测结果，`-o`选项自定义相关参数来对yaml配置文件中的参数进行更新。
```bash
# COCO格式数据集
python test.py --cfg ./config/yolov5.yaml \
       --dataset_dir /home/dpw/deeplearning/coco2017 \
       --dataset_type coco \
       --choice val \
       --result_dir ./result/test \
       --save_image \
       --export_time \
       --print_detection_result \
       -o DetectionModel.engine_type=onnx DetectionModel.engine_model_path=./model_data/yolov5n.onnx DetectionModel.confidence_threshold=0.001 DetectionModel.iou_threshold=0.5

# VOC格式数据集
python test.py --cfg ./config/yolov5.yaml \
       --dataset_dir /home/dpw/deeplearning/coco2017_voc \
       --dataset_type voc \
       --choice val \
       --result_dir ./result/test \
       --save_image \
       --export_time \
       --print_detection_result \
       -o DetectionModel.engine_type=onnx DetectionModel.engine_model_path=./model_data/yolov5n.onnx DetectionModel.confidence_threshold=0.001 DetectionModel.iou_threshold=0.5
```

## 3.3 图像视频预标注成VOC数据集
`imageset2voc_dataset.py`主要功能是对图像集进行预标注生成VOC数据集，`video2voc_dataset.py`主要功能是对视频(集)进行预标注生成VOC数据集。其中命令行参数含义如下：`cfg`为模型参数配置文件路径，`imageset`为图像集文件夹路径，`result_dir`为评估结果文件夹路径，`num_threads`为线程数，`interval`为视频抽帧频率，若为-1代表逐帧检测，`print_detection_result`代表是否打印检测结果，`-o`选项自定义相关参数来对yaml配置文件中的参数进行更新。
```bash
# 视频转VOC数据集,并进行预标注
python video2voc_dataset.py --cfg ./config/yolov5.yaml \
       --video ./video/ \
       --result_dir ./result_dir/voc_dataset \
       --num_threads 4 \
       --interval -1 \
       --print_detection_result \
       -o DetectionModel.model_type=yolov5 DetectionModel.engine_model_path=./model_data/yolov5s.trt.fp16 DetectionModel.confidence_threshold=0.5 DetectionModel.iou_threshold=0.5

# 图像集转VOC数据集,并进行预标注
python imageset2voc_dataset.py --cfg ./config/yolov5.yaml \
       --imageset ./image/coco_calib/ \
       --result_dir ./result_dir/voc_dataset \
       --num_threads 4 \
       -o DetectionModel.model_type=yolov5 DetectionModel.engine_model_path=./model_data/yolov5s.trt.fp16 DetectionModel.confidence_threshold=0.5 DetectionModel.iou_threshold=0.5
```

## 3.4 图像视频预标注成Labelme数据集
`imageset2labelme_dataset.py`主要功能是对图像集进行预标注生成Labelme数据集，`video2labelme_dataset.py`主要功能是对视频(集)进行预标注生成Labelme数据集。其中命令行参数含义如下：`cfg`为模型参数配置文件路径，`imageset`为图像集文件夹路径，`result_dir`为评估结果文件夹路径，`num_threads`为线程数，`interval`为视频抽帧频率，若为-1代表逐帧检测，`print_detection_result`代表是否打印检测结果，`-o`选项自定义相关参数来对yaml配置文件中的参数进行更新。
```bash
# 视频转VOC数据集,并进行预标注
python video2labelme_dataset.py --cfg ./config/yolov5.yaml \
       --video ./video/ \
       --result_dir ./result_dir/labelme_dataset \
       --num_threads 4 \
       --interval -1 \
       --print_detection_result \
       -o DetectionModel.model_type=yolov5 DetectionModel.engine_model_path=./model_data/yolov5s.trt.fp16 DetectionModel.confidence_threshold=0.5 DetectionModel.iou_threshold=0.5

# 图像集转VOC数据集,并进行预标注
python imageset2labelme_dataset.py --cfg ./config/yolov5.yaml \
       --imageset ./image/coco_calib/ \
       --result_dir ./result_dir/labelme_dataset \
       --num_threads 4 \
       -o DetectionModel.model_type=yolov5 DetectionModel.engine_model_path=./model_data/yolov5s.trt.fp16 DetectionModel.confidence_threshold=0.5 DetectionModel.iou_threshold=0.5
```