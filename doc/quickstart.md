# Quick Start
这是YOLO_Deploy项目快速开始的文档

---
# 环境安装
由于ONNX格式是众多推理框架的所兼容中间格式，因此首先安装ONNX相关库． 根据`requirements.txt`安装ONNX相关环境，命令如下:
```bash
conda create deploy python=3.8
conda activate deploy
pip install requirements.txt
```
**上述环境仅支持ONNX和ONNXRuntime部署**，对于其他框架的安装需要根据实际情况进行安装．目前支持的推理框架如下:
- ONNX/ONNXRuntime
- TensorRT
- ...

## TensorRT
在[TensorRT官网](https://developer.nvidia.com/tensorrt)下载所需安装包并解压，在这里以TensorRT7.2.2.3安装举例，安装流程如下：
```bash
# 解压安装包
sudo chmod 777 /path/to/TensorRT-7.2.2.3.Ubuntu-18.04.x86_64-gnu.cuda-11.1.cudnn8.0.tar.gz
sudo tar -zxvf /path/to/TensorRT-7.2.2.3.Ubuntu-18.04.x86_64-gnu.cuda-11.1.cudnn8.0.tar.gz
sudo chmod 777 -R /path/to/TensorRT-7.2.2.3/
# 安装TensorRT
pip install cython pycuda==2019.1
cd /path/to/TensorRT-7.2.2.3/
cd python 
pip install tensorrt-7.2.2.3-cp38-none-linux_x86_64.whl
cd ..
cd graphsurgeon
pip install graphsurgeon-0.4.5-py2.py3-none-any.whl
cd ..
cd onnx_graphsurgeon
pip install onnx_graphsurgeon-0.2.6-py2.py3-none-any.whl
cd ..
cd uff
pip install uff-0.6.9-py2.py3-none-any.whl
```
安装完成后在再`~/.bashrc`中设置TensorRT的环境变量，命令如下：
```bash
sudo vim  ~/.bashrc
# 下面命令假假如~/.bashrc文件文件中，需要根据自身需要调整TensorRT文件夹的绝对路径
export TENSORRT_ROOT=/path/to/TensorRT-7.2.2.3/
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/path/to/TensorRT-7.2.2.3/lib

# 然后使配置文件生效
source ~/.bashrc
```

---
# 部署
部署阶段，目前支持检测图像(集)和视频(集)，支持在VOC和COCO格式的目标检测数据集上评估检测模型性能，对图片集和视频进行预标生成VOC数据集格式标签．
## 检测图像和视频
在`detect.py`中可以根据`-o`选项自定义相关参数来对yaml配置文件中的参数进行更新．
```bash
# 图片
python detect.py --cfg ./config/detection.yaml \
       --source ./image/ \
       --result_dir ./result/image/yolov5s/confidence_threshold=0.3_iou_threshold=0.5 \
       -o DetectionModel.model_type=yolov5 DetectionModel.model_path=./model_data/yolov5s.onnx DetectionModel.class_names_path=./model_data/coco_names.txt DetectionModel.input_shape=[1,3,640,640] DetectionModel.confidence_threshold=0.3 DetectionModel.iou_threshold=0.5

# 视频,逐帧检测
python detect.py --cfg ./config/detection.yaml \
       --source ./video/1.mp4 \
       --result_dir ./result/video/yolov5s/confidence_threshold=0.3_iou_threshold=0.5 \
       --interval -1
       -o DetectionModel.model_type=yolov5 DetectionModel.model_path=./model_data/yolov5s.onnx DetectionModel.class_names_path=./model_data/coco_names.txt DetectionModel.input_shape=[1,3,640,640] DetectionModel.confidence_threshold=0.3 DetectionModel.iou_threshold=0.5

# 视频,隔秒检测
python detect.py --cfg ./config/detection.yaml \
       --source ./video/1.mp4 \
       --result_dir ./result/video/yolov5s/confidence_threshold=0.3_iou_threshold=0.5 \
       --interval 1
       -o DetectionModel.model_type=yolov5 DetectionModel.model_path=./model_data/yolov5s.onnx DetectionModel.class_names_path=./model_data/coco_names.txt DetectionModel.input_shape=[1,3,640,640] DetectionModel.confidence_threshold=0.3 DetectionModel.iou_threshold=0.5

# 图片或者图片文件夹或者视频或者视频文件夹或者任意组合
python detect.py --cfg ./config/detection.yaml \
       --source ./video/ \
       --result_dir ./result/image_video/yolov5s/confidence_threshold=0.3_iou_threshold=0.5 \
       --interval -1
       -o DetectionModel.model_type=yolov5 DetectionModel.model_path=./model_data/yolov5s.onnx DetectionModel.class_names_path=./model_data/coco_names.txt DetectionModel.input_shape=[1,3,640,640] DetectionModel.confidence_threshold=0.3 DetectionModel.iou_threshold=0.5
```

## 评估模型性能
在`test.py`中可以根据`-o`选项自定义相关参数来对yaml配置文件中的参数进行更新．
```bash
# COCO格式数据集
python test.py --cfg ./config/detection.yaml \
       --dataset_dir ../COC2017 \
       --result_dir ./result/COC2017/yolov5s \
       --choice 'val'  \
       --dataset_type 'coco' \
       --save_image \
       -o DetectionModel.model_type=yolov5 DetectionModel.model_path=./model_data/yolov5s.onnx DetectionModel.class_names_path=./model_data/coco_names.txt DetectionModel.input_shape=[1,3,640,640] DetectionModel.confidence_threshold=0.001 DetectionModel.iou_threshold=0.5

# VOC格式数据集
python test.py --cfg ./config/detection.yaml \
       --dataset_dir ../VOC2007 \
       --result_dir ./result/VOC2007/yolov5s \
       --choice 'val'  \
       --dataset_type 'voc' \
       --save_image \
       -o DetectionModel.model_type=yolov5 DetectionModel.model_path=./model_data/yolov5s.onnx DetectionModel.class_names_path=./model_data/voc_names.txt DetectionModel.input_shape=[1,3,640,640] DetectionModel.confidence_threshold=0.001 DetectionModel.iou_threshold=0.5
```

## 图像视频预标注
在`video2voc_dataset.py`和`imageset2voc_dataset.py`中可以根据`-o`选项自定义相关参数来对yaml配置文件中的参数进行更新．
```bash
# 视频转VOC数据集,并进行预标注
python video2voc_dataset.py --cfg ./config/detection.yaml \
       --video ./video/ \
       --result_dir ./result/video \
       --interval 1 \
       --num_threads 1 \
       -o DetectionModel.model_type=yolov5 DetectionModel.model_path=./model_data/yolov5s.onnx DetectionModel.class_names_path=./model_data/coco_names.txt DetectionModel.input_shape=[1,3,640,640] DetectionModel.confidence_threshold=0.3 DetectionModel.iou_threshold=0.5

# 图像集转VOC数据集,并进行预标注
python imageset2voc_dataset.py --cfg ./config/detection.yaml \
       --imageset ./image/ \
       --result_dir ./result \
       --num_threads 4 \
       --confidence_threshold 0.1
       -o DetectionModel.model_type=yolov5 DetectionModel.model_path=./model_data/yolov5s.onnx DetectionModel.class_names_path= DetectionModel.input_shape=[1,3,640,640] DetectionModel.confidence_threshold=0.001 DetectionModel.iou_threshold=0.5
```

## VOC转COCO
```bash
python voc2coco.py --voc_dataset_dir ../VOC2007 \
       --coco_dataset_dir ../VOC2007-COCO \
       --class_name_path ./model_data/voc_names.txt \
       --choices "train" "val" --save_image
```
