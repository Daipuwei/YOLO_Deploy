# onnx2tensorrt
## fp32
python onnx2tensorrt.py --onnx_model_path ./model_data/yolov5s.onnx \
       --tensorrt_model_path ./model_data/yolov5s.trt \
       --input_shape 1 3 640 640 \
       --model_type yolov5 \
       --mode fp32

## fp16
python onnx2tensorrt.py --onnx_model_path ./model_data/yolov5s.onnx \
       --tensorrt_model_path ./model_data/yolov5s.trt \
       --input_shape 1 3 640 640 \
       --model_type yolov5 \
       --mode fp16

## int8
python onnx2tensorrt.py --onnx_model_path ./model_data/yolov5s.onnx \
       --tensorrt_model_path ./model_data/yolov5s.trt \
       --input_shape 1 3 640 640 \
       --model_type yolov5 \
       --mode int8 \
       --calibrator_image_dir ./image/coco_calib/ \
       --calibrator_table_path ./model_data/yolov5s_coco_calibrator_table.cache \
       --data_type float32

# 检测图像(集)或视频(集)
python detect.py --cfg ./config/detection.yaml \
       --source ./image/coco_calib/ \
       --result_dir ./result/coco_calib/yolov5s.trt.fp16 \
       -o DetectionModel.model_type=yolov5 DetectionModel.engine_model_path=./model_data/yolov5s.trt.fp16 DetectionModel.confidence_threshold=0.5

# 检测视频(集)， 逐帧检测
python detect.py --cfg ./config/detection.yaml \
       --source ./video/1.dav \
       --result_dir ./result/image_video \
       --interval -1 \
       -o DetectionModel.model_type=yolov5 DetectionModel.engine_model_path=./model_data/yolov5s.trt.fp16 DetectionModel.confidence_threshold=0.1

# 检测视频(集)，隔秒检测
python detect.py --cfg ./config/detection.yaml \
       --source ./video/1.dav \
       --result_dir ./result/image_video \
       --interval 1 \
       -o DetectionModel.model_type=yolov5 DetectionModel.engine_model_path=./model_data/yolov5s.trt.fp16 DetectionModel.confidence_threshold=0.1


# 测试模型性能
python test.py --cfg ./config/detection.yaml \
       --dataset_dir ./coco2014 \
       --dataset_type coco \
       --choice test \
       --result_dir ./result/test \
       --save_image \
       --export_time \
       -o DetectionModel.model_type=yolov5 DetectionModel.engine_model_path=./model_data/yolov5s.trt.fp16 DetectionModel.confidence_threshold=0.001 DetectionModel.iou_threshold=0.5

# 图像集预标注成VOC数据集
python imageset2voc_dataset.py --cfg ./config/detection.yaml \
       --imageset ./image/coco_calib/ \
       --result_dir ./result/voc_dataset \
       --num_threads 4 \
       -o DetectionModel.model_type=yolov5 DetectionModel.engine_model_path=./model_data/yolov5s.trt.fp16 DetectionModel.confidence_threshold=0.5 DetectionModel.iou_threshold=0.5

# 图像集预标注成labelme数据集
python imageset2labelme_dataset.py --cfg ./config/detection.yaml \
       --imageset ./image/coco_calib/ \
       --result_dir ./result/labelme_dataset \
       --num_threads 4 \
       -o DetectionModel.model_type=yolov5 DetectionModel.engine_model_path=./model_data/yolov5s.trt.fp16 DetectionModel.confidence_threshold=0.5 DetectionModel.iou_threshold=0.5


# 视频集预标注成VOC数据集
python video2voc_dataset.py --cfg ./config/detection.yaml \
       --video ./video/111 \
       --result_dir ./result/voc_dataset \
       --num_threads 4 \
       --print_detection_result \
       -o DetectionModel.model_type=yolov5 DetectionModel.engine_model_path=./model_data/yolov5s.trt.fp16 DetectionModel.confidence_threshold=0.5 DetectionModel.iou_threshold=0.5

# 视频集预标注成labelme数据集
python video2labelme_dataset.py --cfg ./config/detection.yaml \
       --video ./video/111 \
       --result_dir ./result/labelme_dataset \
       --num_threads 4 \
       --print_detection_result \
       -o DetectionModel.model_type=yolov5 DetectionModel.engine_model_path=./model_data/yolov5s.trt.fp16 DetectionModel.confidence_threshold=0.5 DetectionModel.iou_threshold=0.5
