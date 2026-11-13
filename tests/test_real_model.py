# -*- coding: utf-8 -*-
"""
    真实模型端到端测试（可选）：模型文件不存在时自动跳过。
    用于验证 真实ONNX/TRT引擎 + yolo26(DFL家族) 全链路装配与推理。

    模型约定（本仓库自定义导出）：
    - 输入 [1,960,960,3] NHWC，div255已融入模型（无归一化预处理）；
    - 输出 [1,N,4+nc+1]：scores_sum在第4列（first布局），类别最大分打分。

    可通过环境变量 REAL_MODEL_ONNX / REAL_MODEL_TRT / REAL_MODEL_CLASSES
    指定模型与类别名文件路径。
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_ONNX = ("/data/Milesight/code_deploy/YOLO_Deploy/model_data/"
                "yolo26s_vca+fish_3520x3520_bs1_960x960_nhwc_div255_with_score_sum_singleoutput.onnx")
DEFAULT_TRT = DEFAULT_ONNX[:-len(".onnx")] + ".trt.fp16"
DEFAULT_CLASSES = "/data/Milesight/code_deploy/YOLO_Deploy/model_data/fish_names.txt"


def _have(path):
    return path and os.path.exists(path)


def test_real_model_onnxruntime():
    onnx = os.environ.get("REAL_MODEL_ONNX", DEFAULT_ONNX)
    classes = os.environ.get("REAL_MODEL_CLASSES", DEFAULT_CLASSES)
    if not (_have(onnx) and _have(classes)):
        print("SKIP test_real_model_onnxruntime（模型文件不存在）")
        return
    import cv2
    from detection_track_toolkit.detection_model import build_model
    from detection_track_toolkit.utils import init_logger
    cfg = {"DetectionModel": {
        "model_type": "yolo26", "engine_type": "onnxruntime",
        "engine_model_path": onnx, "class_name_path": classes,
        "confidence_threshold": 0.3, "iou_threshold": 0.45,
        "is_normalize": False},
        "log_path": "/tmp/real_model_ort.log"}
    model = build_model(init_logger(cfg), cfg["DetectionModel"])
    frame = np.zeros((960, 960, 3), dtype=np.uint8)
    cv2.rectangle(frame, (200, 200), (400, 400), (255, 255, 255), -1)
    preds = model.detect(frame)
    assert isinstance(preds, list), preds      # 检测结果（可为空，链路必须通）
    model.release()


def test_real_model_tensorrt():
    trt_path = os.environ.get("REAL_MODEL_TRT", DEFAULT_TRT)
    classes = os.environ.get("REAL_MODEL_CLASSES", DEFAULT_CLASSES)
    if not (_have(trt_path) and _have(classes)):
        print("SKIP test_real_model_tensorrt（引擎文件不存在）")
        return
    import cv2
    from detection_track_toolkit.detection_model import build_model
    from detection_track_toolkit.utils import init_logger
    cfg = {"DetectionModel": {
        "model_type": "yolo26", "engine_type": "tensorrt",
        "engine_model_path": trt_path, "class_name_path": classes,
        "confidence_threshold": 0.3, "iou_threshold": 0.45,
        "is_normalize": False},
        "log_path": "/tmp/real_model_trt.log"}
    model = build_model(init_logger(cfg), cfg["DetectionModel"])
    frame = np.zeros((960, 960, 3), dtype=np.uint8)
    cv2.rectangle(frame, (200, 200), (400, 400), (255, 255, 255), -1)
    preds = model.detect(frame)
    assert isinstance(preds, list), preds
    model.release()


def _run_all():
    tests = [(k, v) for k, v in list(globals().items())
             if k.startswith("test_") and callable(v)]
    fails = 0
    for name, fn in sorted(tests):
        try:
            fn()
            print("PASS", name)
        except Exception as e:
            fails += 1
            print("FAIL", name, "->", type(e).__name__, str(e)[:200])
    print("=" * 40, "fails:", fails)
    return fails


if __name__ == "__main__":
    sys.exit(1 if _run_all() else 0)
