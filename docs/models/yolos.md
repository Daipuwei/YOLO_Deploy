# yolos
这是yolos的多框架Python部署说明,目前已支持`ONNXRuntime`等框架。

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

|    模型名称     |   分辨率   | mAP@0.5 | mAP@0.5：0.95 | 预处理时间(ms) | 前向推理时间(ms) | 后处理时间(ms) | 推理时间(ms) |
|:-----------:|:-------:|:-------:|:------------:|:---------:|:----------:|:---------:|:--------:|
| yolos_base  | 512x512 |  42.36  |    28.37     |  5.5806   |  28.5160   |  0.2342   | 34.3308  |
| yolos_s_dWr | 512x512 |  38.38  |    24.81     |  5.2931   |  12.7864   |  0.2702   | 18.3497  |
|   yolos_s   | 512x512 |  37.42  |    23.81     |  5.2818   |  12.0771   |  0.2642   | 17.6232  |
|  yolos_ti   | 512x512 |  33.09  |    19.91     |  5.3804   |   8.2221   |  0.3000   | 13.9025  |
|             |         |         |              |           |            |           |          |
| yolos_base  | 640x640 |  44.35  |    29.88     |  8.8529   |  50.3685   |  0.2135   | 59.4350  |
| yolos_s_dWr | 640x640 |  40.14  |    26.22     |  3.1832   |  30.4568   |  1.3228   | 34.9628  |
|   yolos_s   | 640x640 |  39.31  |    25.40     |  8.7285   |  20.6583   |  0.2540   | 29.6408  |
|  yolos_ti   | 640x640 |  33.30  |    20.38     |  8.9315   |  12.3260   |  0.2934   | 21.5509  |
|             |         |         |              |           |            |           |          |
| yolos_base  | 800x800 |  45.47  |    30.84     |  13.7905  |  98.8693   |  0.2121   | 112.8719 |
| yolos_s_dWr | 800x800 |  41.61  |    27.39     |  13.8959  |  43.3910   |  0.2526   | 57.5395  |
|   yolos_s   | 800x800 |  40.49  |    26.16     |  13.9963  |  40.0241   |  0.2510   | 54.2714  |
|  yolos_ti   | 800x800 |  34.22  |    20.64     |  13.8517  |  20.2883   |  0.2864   | 34.4264  |

## 1.2 TensorRT
**TensorRT模型转换成功，但是推理严重掉点，待后续排查修复bug。**

---
# 2. 模型转换
## 2.1 Pytorch2ONNX
首先下载[hustvl/YOLOS](https://github.com/hustvl/YOLOS)。
```bash
git clone https://github.com/hustvl/YOLOS
```
在官网下载对应模型保存。接着然后`models/detector.py`代码中`Detecor`类中添加如下代码。
```python
def forward_dummy(self, samples):
    # import pdb;pdb.set_trace()
    x = self.backbone(samples)
    # x = x[:, 1:,:]
    outputs_class = self.class_embed(x)
    outputs_coord = self.bbox_embed(x).sigmoid()

    prob = F.softmax(outputs_class, -1)
    scores, labels = prob[..., :-1].max(-1)

    # convert to [x0, y0, x1, y1] format
    boxes = box_ops.box_cxcywh_to_xyxy(outputs_coord)
    # and from relative [0, 1] to absolute [0, height] coordinates
    output = torch.cat((boxes, scores.unsqueeze(-1), labels.unsqueeze(-1)), -1)
    #output = torch.cat((outputs_coord, scores.unsqueeze(-1), labels.unsqueeze(-1)), -1)
    #output = torch.cat((boxes, prob[..., :-1]), -1)
    return output
```
并将`util/box_ops.py`中的`box_cxcywh_to_xyxy`函数用下面代码替代
```python
def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.split((1,1,1,1), -1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.cat(b, dim=-1)
```
最后将项目中的`tools/yolos_pytorch2onnx.py`复制到yolos项目中，并重命名为`pytorch2onnx.py`,用于生成yolos的onnx模型。
```bash
# yolos-tiny
python pytorch2onnx.py --backbone_name tiny \
    --device cpu \
    --init_pe_size 800 1333 \
    --resume ./yolos_ti.pth \
    --height 800 \
    --width 800 \
    --onnx_model_path ./yolos_ti.onnx \
    --opset 11


# yolos-s
python pytorch2onnx.py --backbone_name small \
    --device cpu \
    --init_pe_size 512 864 \
    --mid_pe_size 512 864 \
    --resume ./yolos_s_300_pre.pth \
    --height 800 \
    --width 800 \
    --onnx_model_path ./yolos_s_300_pre.onnx \
    --opset 11

# yolos-s-dwr
python pytorch2onnx.py --backbone_name small_dWr \
    --device cpu \
    --init_pe_size 512 864 \
    --mid_pe_size 512 864 \
    --resume ./yolos_s_dWr.pth \
    --height 800 \
    --width 800 \
    --onnx_model_path ./yolos_s_dWr.onnx \
    --opset 11

# yolos-b
python pytorch2onnx.py --backbone_name base \
    --device cpu \
    --init_pe_size 800 1344 \
    --mid_pe_size 800 1344 \
    --resume ./yolos_base.pth \
    --height 800 \
    --width 800 \
    --onnx_model_path ./yolos_base.onnx \
    --opset 11
```

## 2.2 ONNX2TensorRT
TensorRT模型分成fp32、fp16和int8三种模型，以yolos_base为例，转换命令如下。其中命令行参数含义如下：`onnx_model_path`为ONNX模型路径，`tensorrt_model_path`为TensorRT模型路径，`input_shape`为onnx模型输入尺度，`mode`代表TensorRT模型模式，只能选择fp32、fp16和int8三者之一，`model_type`代表模型类型，用于初始化指定模型的TensorRT模型int8校准类，`calibrator_image_dir`代表TensorRT模型int8校准集路径，`calibrator_table_path`代表TensorRT模型int8校准表路径，`data_type`代表int8校准类当中数据张量类型，默认为float32。
```bash
# fp32
python onnx2tensorrt.py --onnx_model_path ./model_data/yolos_base_800x800.onnx \
       --tensorrt_model_path ./model_data/yolos_base.trt \
       --input_shape 1 3 800 800 \
       --mode fp32

# fp16
python onnx2tensorrt.py --onnx_model_path ./model_data/yolos_base_800x800.onnx \
       --tensorrt_model_path ./model_data/yolos_base_800x800.trt \
       --input_shape 1 3 640 640 \
       --mode fp16

# int8
python onnx2tensorrt.py --onnx_model_path ./model_data/yolos_base_800x800.onnx \
       --tensorrt_model_path ./model_data/yolos_base_800x800.trt \
       --input_shape 1 3 640 640 \
       --mode int8 \
       --model_type yolos \
       --calibrator_image_dir ./image/coco_calib/ \
       --calibrator_table_path ./model_data/yolos_base_800x800_coco_calibrator_table.cache \
       --data_type float32
```

---
# 3. 功能介绍
## 3.1 检测图像和视频
`detect.py`主要功能为检测图像(集)和视频(集)。其中命令行参数含义如下：`cfg`为模型参数配置文件路径，`source`为待检测的图像(集)或视频(集)路径，`result_dir`为检测结果文件夹路径，`interval`为视频抽帧频率，若为-1代表逐帧检测，`num_threads`为线程数，`print_detection_result`代表是否打印检测结果，`-o`选项自定义相关参数来对yaml配置文件中的参数进行更新。
```bash
# 检测图片(集)
python detect.py --cfg ./config/yolos.yaml \
       --source ./image/coco_calib/ \
       --result_dir ./result/image_video/ \
       -o DetectionModel.engine_model_path=./model_data/yolos_base_800x800.onnx DetectionModel.confidence_threshold=0.5

# 检测视频(集)， 逐帧检测
python detect.py --cfg ./config/yolos.yaml \
       --source ./video/1.dav \
       --result_dir ./result/image_video/ \
       --interval -1 \
       -o DetectionModel.model_type=yolos DetectionModel.engine_model_path=./model_data/yolos_base_800x800.onnx DetectionModel.confidence_threshold=0.1

# 检测视频(集)，隔秒检测
python detect.py --cfg ./config/yolos.yaml \
       --source ./video/1.dav \
       --result_dir ./result/image_video/ \
       --interval 1 \
       -o DetectionModel.model_type=yolos DetectionModel.engine_model_path=./model_data/yolos_base_800x800.onnx DetectionModel.confidence_threshold=0.1

# 图片或者图片文件夹或者视频或者视频文件夹或者任意组合
python detect.py --cfg ./config/yolos.yaml \
       --source ./image_video/ \
       --result_dir ./result/image_video/ \
       --interval -1 \
       -o DetectionModel.model_type=yolos DetectionModel.engine_model_path=./model_data/yolos_base_800x800.onnx DetectionModel.confidence_threshold=0.1
```

## 3.2 评估模型性能
`test.py`主要功能为评估模型在VOC格式数据集或者COCO格式数据集上的性能。其中命令行参数含义如下：`cfg`为模型参数配置文件路径，`dataset_dir`为数据集路径，`dataset_type`为数据集类型，候选值为voc和coco，`choice`为数据集子集类型,`result_dir`为评估结果文件夹路径，`save_image`代表是否保存检测图像，`export_time`代表是否输出推理时间选项，`print_detection_result`代表是否打印检测结果，`-o`选项自定义相关参数来对yaml配置文件中的参数进行更新。
```bash
# COCO格式数据集
python test.py --cfg ./config/yolos.yaml \
       --dataset_dir /home/dpw/deeplearning/coco2017 \
       --dataset_type coco \
       --choice val \
       --result_dir ./result/test \
       --save_image \
       --export_time \
       --print_detection_result \
       -o DetectionModel.engine_model_path=./model_data/yolos_base_800x800.onnx DetectionModel.confidence_threshold=0.001 DetectionModel.iou_threshold=0.5

# VOC格式数据集
python test.py --cfg ./config/yolos.yaml \
       --dataset_dir /home/dpw/deeplearning/coco2017_voc \
       --dataset_type voc \
       --choice val \
       --result_dir ./result/test \
       --save_image \
       --export_time \
       --print_detection_result \
       -o DetectionModel.engine_model_path=./model_data/yolos_base_800x800.onnx DetectionModel.confidence_threshold=0.001 DetectionModel.iou_threshold=0.5
```

## 3.3 图像视频预标注成VOC数据集
`imageset2voc_dataset.py`主要功能是对图像集进行预标注生成VOC数据集，`video2voc_dataset.py`主要功能是对视频(集)进行预标注生成VOC数据集。其中命令行参数含义如下：`cfg`为模型参数配置文件路径，`imageset`为图像集文件夹路径，`result_dir`为评估结果文件夹路径，`num_threads`为线程数，`interval`为视频抽帧频率，若为-1代表逐帧检测，`print_detection_result`代表是否打印检测结果，`-o`选项自定义相关参数来对yaml配置文件中的参数进行更新。
```bash
# 视频转VOC数据集,并进行预标注
python video2voc_dataset.py --cfg ./config/yolos.yaml \
       --video ./video/ \
       --result_dir ./result_dir/voc_dataset \
       --num_threads 4 \
       --interval -1 \
       --print_detection_result \
       -o DetectionModel.model_type=yolos DetectionModel.engine_model_path=./model_data/yolos_base_800x800.onnx DetectionModel.confidence_threshold=0.5 DetectionModel.iou_threshold=0.5

# 图像集转VOC数据集,并进行预标注
python imageset2voc_dataset.py --cfg ./config/yolos.yaml \
       --imageset ./image/coco_calib/ \
       --result_dir ./result_dir/voc_dataset \
       --num_threads 4 \
       -o DetectionModel.model_type=yolos DetectionModel.engine_model_path=./model_data/yolos_base_800x800.onnx DetectionModel.confidence_threshold=0.5 DetectionModel.iou_threshold=0.5
```

## 3.4 图像视频预标注成Labelme数据集
`imageset2labelme_dataset.py`主要功能是对图像集进行预标注生成Labelme数据集，`video2labelme_dataset.py`主要功能是对视频(集)进行预标注生成Labelme数据集。其中命令行参数含义如下：`cfg`为模型参数配置文件路径，`imageset`为图像集文件夹路径，`result_dir`为评估结果文件夹路径，`num_threads`为线程数，`interval`为视频抽帧频率，若为-1代表逐帧检测，`print_detection_result`代表是否打印检测结果，`-o`选项自定义相关参数来对yaml配置文件中的参数进行更新。
```bash
# 视频转VOC数据集,并进行预标注
python video2labelme_dataset.py --cfg ./config/yolos.yaml \
       --video ./video/ \
       --result_dir ./result_dir/labelme_dataset \
       --num_threads 4 \
       --interval -1 \
       --print_detection_result \
       -o DetectionModel.model_type=yolos DetectionModel.engine_model_path=./model_data/yolos_base_800x800.onnx DetectionModel.confidence_threshold=0.5 DetectionModel.iou_threshold=0.5

# 图像集转VOC数据集,并进行预标注
python imageset2labelme_dataset.py --cfg ./config/yolos.yaml \
       --imageset ./image/coco_calib/ \
       --result_dir ./result_dir/labelme_dataset \
       --num_threads 4 \
       -o DetectionModel.model_type=yolos DetectionModel.engine_model_path=./model_data/yolos_base_800x800.onnx DetectionModel.confidence_threshold=0.5 DetectionModel.iou_threshold=0.5
```