# -*- coding: utf-8 -*-
"""
    端到端对拍测试：FakeEngine输出与GT完全一致的固定检测框，
    验证 数据集解析->gt坐标转换->模型检测->JSON落盘->COCOeval/YOLO指标 整条链路。
    GT与预测一致时 mAP@0.5 与 mAP@0.5:0.95 均应为1.0，
    若gt坐标约定（xywh/x1y1x2y2）处理错误，指标会显著偏离1.0。
"""

import os
import sys
import json
import shutil
import tempfile

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from detection_track_toolkit.engine.base_engine import BaseEngine
from detection_track_toolkit.engine import ENGINE_REGISTRY
from detection_track_toolkit.detection_model import build_model
from detection_track_toolkit.utils import init_logger
from detection_track_toolkit.utils import eval_coco
from detection_track_toolkit.utils import save_yaml
from detection_track_toolkit.utils import load_yaml

# 固定检测框（模型输入尺度与原图尺度一致，gain=1）：[x1,y1,x2,y2]=[10,10,40,40]，类别0
GT_BOX = (10, 10, 40, 40)
IMG = np.zeros((64, 64, 3), dtype=np.uint8)


class FakeEngine(BaseEngine):
    """测试引擎：无论输入是什么，恒输出与GT一致的检测框"""

    def __init__(self, logger, engine_model_path, **kwargs):
        super(FakeEngine, self).__init__(logger, engine_model_path, **kwargs)
        self.input_names = ["images"]
        self.input_shapes = [(1, 3, 64, 64)]
        self.output_names = ["output"]
        self.output_shapes = [(1, 1, 7)]
        self.is_nchw = self.judge_is_nchw(self.input_shapes)

    def get_is_nchw(self):
        return self.is_nchw

    def inference(self, input_tensor):
        batch = input_tensor[0].shape[0]
        # YOLOX输出布局(1,1,5+nc): [x1,y1,x2,y2,obj,cls0,cls1]
        out = np.zeros((batch, 1, 7), dtype=np.float32)
        out[:, 0, 0:4] = GT_BOX
        out[:, 0, 4] = 0.9
        out[:, 0, 5] = 0.95
        out[:, 0, 6] = 0.1
        return [out]


@ENGINE_REGISTRY.register()
def fake(logger, cfg, **kwargs):
    return FakeEngine(logger, cfg["engine_model_path"], **kwargs)


class FakeEngineBatch2(FakeEngine):
    """批量=2的假引擎：用于覆盖bs>1尾批单图的评测解包路径"""

    def __init__(self, logger, engine_model_path, **kwargs):
        super(FakeEngineBatch2, self).__init__(logger, engine_model_path, **kwargs)
        self.input_shapes = [(2, 3, 64, 64)]
        self.output_shapes = [(2, 1, 7)]


@ENGINE_REGISTRY.register()
def fake_batch2(logger, cfg, **kwargs):
    return FakeEngineBatch2(logger, cfg["engine_model_path"], **kwargs)


def _make_voc_dataset(root, n_images=2):
    os.makedirs(os.path.join(root, "Annotations"))
    os.makedirs(os.path.join(root, "JPEGImages"))
    os.makedirs(os.path.join(root, "ImageSets", "Main"))
    ids = []
    for i in range(n_images):
        fid = "%04d" % (i + 1)
        ids.append(fid)
        cv2.imwrite(os.path.join(root, "JPEGImages", fid + ".jpg"), IMG)
        with open(os.path.join(root, "Annotations", fid + ".xml"), "w") as f:
            f.write('''<annotation><object><name>person</name><bndbox>
            <xmin>{0}</xmin><ymin>{1}</ymin><xmax>{2}</xmax><ymax>{3}</ymax>
            </bndbox></object></annotation>'''.format(*GT_BOX))
    with open(os.path.join(root, "ImageSets", "Main", "val.txt"), "w") as f:
        f.write("\n".join(ids) + "\n")


def test_detect_single_image():
    tmp = tempfile.mkdtemp()
    try:
        classes_path = os.path.join(tmp, "classes.txt")
        with open(classes_path, "w") as f:
            f.write("person\ncar\n")
        model_path = os.path.join(tmp, "fake.onnx")
        open(model_path, "w").close()
        cfg = {"DetectionModel": {
            "model_type": "yolox", "engine_type": "fake",
            "engine_model_path": model_path, "class_name_path": classes_path,
            "confidence_threshold": 0.2, "iou_threshold": 0.45}}
        logger = init_logger({"log_path": os.path.join(tmp, "t.log")})
        model = build_model(logger, cfg["DetectionModel"])
        preds = model.detect(IMG.copy(), export_time=True)
        preds, times = preds
        assert len(preds) == 1, preds
        p = preds[0]
        assert p["bbox"] == list(GT_BOX), p
        assert p["cls_name"] == "person" and p["cls_id"] == 0, p
        assert abs(p["score"] - 0.95 * 0.9) < 1e-3, p
        assert set(times.keys()) == {"preprocess_time", "inference_time",
                                     "postprocess_time", "detect_time"}
        model.release()
    finally:
        shutil.rmtree(tmp)


def test_detect_pads_partial_batch():
    """不满批自动补零图凑满（回归：数据集尾批不满batchsize/视频单帧检测在
    batch>1固定批量引擎上曾直接崩溃——TRT元素数校验ValueError/ORT shape错误）：
    引擎必须收到满批张量，后处理只收到真实图像数，结果不含补批行"""
    from detection_track_toolkit.detection_model.base_model import DetectionModel

    batch = 4

    class FixedBatchEngine:
        """固定批量引擎：断言每次收到的张量batch维必须等于batch"""
        def __init__(self):
            self.seen_batches = []

        def inference(self, input_tensor):
            tensor = input_tensor[0]
            self.seen_batches.append(tensor.shape[0])
            assert tensor.shape[0] == batch, \
                "固定批量引擎应收到满批{0}，实际{1}".format(batch, tensor.shape[0])
            return [np.zeros((batch, 1, 7), dtype=np.float32)]

        def release(self):
            pass

    class RecordingPostprocessor:
        """记录后处理收到的image_shapes数量（应为真实图像数而非补批后数量）"""
        confidence_threshold = 0.2
        iou_threshold = 0.45
        class_names = ["person", "car"]

        def __init__(self):
            self.rows_seen = []

        def postprocess(self, outputs, image_shapes, restore_params=None):
            self.rows_seen.append(len(image_shapes))
            # 返回形态与BasePostprocessor.postprocess一致：每图像一个检测字典列表
            return [[{"bbox": list(GT_BOX), "score": 0.9,
                      "cls_id": 0, "cls_name": "person"}]
                    for _ in range(len(image_shapes))]

        def release(self):
            pass

    class IdentityPreprocessor:
        batch_size = batch
        c = 3
        h = 64
        w = 64
        is_nchw = False

        def preprocess(self, images):
            # 返回形态与BasePreprocessor.preprocess一致（三块契约三元组）：
            # 批量张量 + 逐图原始尺度 + 逐图缩放逆变换参数
            num = len(images)
            return (np.zeros((num, self.h, self.w, self.c), dtype=np.float32),
                    np.array([[self.h, self.w]] * num),
                    np.tile(np.array([[1.0, 1.0, 0.0, 0.0]], dtype=np.float32),
                            (num, 1)))

        def release(self):
            pass

    tmp = tempfile.mkdtemp()
    try:
        logger = init_logger({"log_path": os.path.join(tmp, "t.log")})
        engine = FixedBatchEngine()
        post = RecordingPostprocessor()
        model = DetectionModel(logger, engine, IdentityPreprocessor(),
                               post, model_type="yolox")
        # 3张图（尾批不满4）：补零凑满，结果3张
        preds = model.detect([IMG.copy() for _ in range(3)])
        assert len(preds) == 3, preds
        assert engine.seen_batches == [4], engine.seen_batches
        assert post.rows_seen == [3], post.rows_seen
        # 单张图（detect_video单帧场景，batch>1模型；单图返回该图检测列表）
        preds = model.detect(IMG.copy())
        assert len(preds) == 1 and preds[0]["bbox"] == list(GT_BOX), preds
        assert engine.seen_batches == [4, 4], engine.seen_batches
        assert post.rows_seen == [3, 1], post.rows_seen
        # 超过batch_size：仍报错（既有行为不变）
        try:
            model.detect([IMG.copy() for _ in range(5)])
            assert False, "超过batch_size应报错"
        except ValueError:
            pass
        model.release()
    finally:
        shutil.rmtree(tmp)


def test_resolve_config_path_anchoring():
    """配置相对路径解析口径（回归#7：曾一律按进程CWD解析，非仓库根目录
    运行工具脚本时模型/类别名/日志路径全部FileNotFoundError）：
    绝对路径原样；CWD存在时优先CWD（自定义工程兼容）；CWD不存在回退仓库根"""
    from detection_track_toolkit.utils import resolve_config_path, REPO_ROOT
    abs_path = "/tmp/definitely/absolute/x.onnx"
    assert resolve_config_path(abs_path) == abs_path
    # None/空串显式报错（回归：曾静默返回仓库根/抛原生TypeError）
    for bad in (None, ""):
        try:
            resolve_config_path(bad)
            assert False, "None/空串应报错"
        except ValueError:
            pass
    cwd = os.getcwd()
    tmp = tempfile.mkdtemp()
    try:
        os.chdir(tmp)
        # CWD下不存在：回退锚定仓库根
        assert resolve_config_path("./model_data/coco.names") == \
            os.path.join(REPO_ROOT, "model_data", "coco.names")
        # CWD下存在同名相对路径：优先CWD
        os.makedirs(os.path.join(tmp, "model_data"))
        with open(os.path.join(tmp, "model_data", "coco.names"), "w") as f:
            f.write("custom\n")
        assert resolve_config_path("./model_data/coco.names") == \
            os.path.abspath("./model_data/coco.names")
    finally:
        os.chdir(cwd)
        shutil.rmtree(tmp)


def test_model_builds_from_non_repo_cwd():
    """非仓库根CWD下按配置模板相对路径装配模型（回归#7）：
    dflyolo的class_name_path=./model_data/...回退仓库根解析成功；
    yolov5的anchors缺省（./model_data/coco_anchors.txt）不再写死机器
    绝对路径，非仓库根CWD下同样回退仓库根加载成功"""
    from detection_track_toolkit.utils import REPO_ROOT
    cwd = os.getcwd()
    tmp = tempfile.mkdtemp()
    try:
        classes_path = os.path.join(tmp, "classes.txt")
        with open(classes_path, "w") as f:
            f.write("person\ncar\n")
        model_path = os.path.join(tmp, "fake.onnx")
        open(model_path, "w").close()
        logger = init_logger({"log_path": os.path.join(tmp, "t.log")})
        os.chdir(tmp)
        # dflyolo家族：模板式相对class_name_path，tmp下无model_data → 回退仓库根
        cfg_dfl = {"model_type": "yolov8", "engine_type": "fake",
                   "engine_model_path": model_path,
                   "class_name_path": "./model_data/coco.names",
                   "confidence_threshold": 0.2, "iou_threshold": 0.45}
        model = build_model(logger, cfg_dfl)
        with open(os.path.join(REPO_ROOT, "model_data", "coco.names")) as f:
            expected = [line.strip() for line in f if line.strip()]
        assert model.get_class_names() == expected, model.get_class_names()[:5]
        model.release()
        # yolov5：不配anchors_path，缺省应从仓库根model_data加载coco_anchors
        cfg_v5 = {"model_type": "yolov5", "engine_type": "fake",
                  "engine_model_path": model_path,
                  "class_name_path": classes_path,
                  "confidence_threshold": 0.2, "iou_threshold": 0.45}
        model5 = build_model(logger, cfg_v5)
        assert model5.postprocessor.anchors is not None
        assert len(model5.postprocessor.anchors) == 1  # fake引擎单输出=1层
        model5.release()
    finally:
        os.chdir(cwd)
        shutil.rmtree(tmp)


def test_detect_with_center_pad_mode():
    """pad_mode=center端到端：128x64宽图letterbox到64x64（缩放0.5、垂直填充32），
    FakeEngine固定输出模型坐标框[10,10,40,40]：
    - center（pad_h=16）：还原需减偏移 → [20, 0, 80, 48]
    - top_left（缺省，pad_h=0）：直接除gain → [20, 20, 80, 64]
      （y2=80超原图高度64，越界裁剪到64属正确行为）
    （验证配置pad_mode经装配透传到预处理、restore_params全链路生效）"""
    tmp = tempfile.mkdtemp()
    try:
        classes_path = os.path.join(tmp, "classes.txt")
        with open(classes_path, "w") as f:
            f.write("person\ncar\n")
        model_path = os.path.join(tmp, "fake.onnx")
        open(model_path, "w").close()
        logger = init_logger({"log_path": os.path.join(tmp, "t.log")})
        img = np.zeros((64, 128, 3), dtype=np.uint8)  # 宽128高64
        for pad_mode, expected in (("center", [20, 0, 80, 48]),
                                   ("top_left", [20, 20, 80, 64])):
            cfg = {"model_type": "yolox", "engine_type": "fake",
                   "engine_model_path": model_path,
                   "class_name_path": classes_path,
                   "confidence_threshold": 0.2, "iou_threshold": 0.45,
                   "pad_mode": pad_mode}
            model = build_model(logger, cfg)
            assert model.preprocessor.pad_mode == pad_mode
            preds = model.detect(img.copy())
            assert len(preds) == 1, preds
            assert preds[0]["bbox"] == expected, (pad_mode, preds[0])
            model.release()
    finally:
        shutil.rmtree(tmp)


def test_scan_media_files_case_insensitive():
    """媒体文件扫描大小写不敏感（回归：.JPG/.PNG/.MP4大写后缀曾被
    静默跳过）；结果路径保留源目录相对层级（子目录不扁平化拼名）"""
    from tools.detect import scan_media_files
    tmp = tempfile.mkdtemp()
    try:
        for name in ("a.JPG", "b.png", "c.MP4", "d.txt"):
            open(os.path.join(tmp, name), "w").close()
        os.makedirs(os.path.join(tmp, "sub"))
        open(os.path.join(tmp, "sub", "e.PNG"), "w").close()
        image_dir = os.path.join(tmp, "image")
        video_dir = os.path.join(tmp, "video")
        imgs, img_results, vids, vid_results = scan_media_files(tmp, image_dir, video_dir)
        assert len(imgs) == 3, imgs                      # a.JPG/b.png/sub/e.PNG
        assert len(vids) == 1, vids                      # c.MP4
        assert all(x.endswith(".jpg") for x in img_results), img_results
        assert all(x.endswith(".mp4") for x in vid_results), vid_results
        # 结果路径保留源目录层级：sub/e.PNG -> image/sub/e.jpg
        assert os.path.join(image_dir, "sub", "e.jpg") in img_results, img_results
        # 主名保留原主名（回归：曾用小写ext做rfind截主名，a.JPG→a.JP.jpg）
        assert {os.path.basename(x) for x in img_results} == \
            {"a.jpg", "b.jpg", "e.jpg"}, img_results
        assert {os.path.basename(x) for x in vid_results} == {"c.mp4"}
        # 单文件模式同样大小写不敏感
        imgs, _, vids, _ = scan_media_files(os.path.join(tmp, "a.JPG"), image_dir, video_dir)
        assert len(imgs) == 1 and len(vids) == 0
    finally:
        shutil.rmtree(tmp)


def test_scan_media_files_preserves_nested_dirs():
    """结果路径保留源目录层级（回归：曾用下划线扁平化，a/b/x.jpg与
    a_b/x.jpg压平成同名结果文件互相覆盖、静默丢一张）"""
    from tools.detect import scan_media_files
    tmp = tempfile.mkdtemp()
    try:
        os.makedirs(os.path.join(tmp, "a", "b"))
        os.makedirs(os.path.join(tmp, "a_b"))
        for p in ("a/b/x.jpg", "a_b/x.jpg"):
            open(os.path.join(tmp, p), "w").close()
        image_dir = os.path.join(tmp, "image")
        imgs, img_results, vids, _ = scan_media_files(tmp, image_dir, "")
        assert len(imgs) == 2 and len(vids) == 0
        # 两个不同路径映射到两个不同结果路径，互不覆盖
        assert len(set(img_results)) == 2, img_results
        assert os.path.join(image_dir, "a", "b", "x.jpg") in img_results
        assert os.path.join(image_dir, "a_b", "x.jpg") in img_results
    finally:
        shutil.rmtree(tmp)


def test_detect_video_interval_step_clamped():
    """视频抽帧步长下限钳到1（回归：interval*fps<1时int截断为0，
    cnt%0 ZeroDivisionError中断检测）；interval=0.02/fps=25曾必崩"""
    from detection_track_toolkit.detection_model.base_model import DetectionModel

    class _NoopEngine:
        def inference(self, input_tensor):
            return [np.zeros((1, 0, 7), dtype=np.float32)]

        def release(self):
            pass

    class _NoopPreprocessor:
        batch_size, c, h, w, is_nchw = 1, 3, 64, 64, False

        def preprocess(self, images):
            num = len(images)
            return (np.zeros((num, 64, 64, 3), dtype=np.float32),
                    np.array([[64, 64]] * num),
                    np.tile(np.array([[1.0, 1.0, 0.0, 0.0]], dtype=np.float32),
                            (num, 1)))

        def release(self):
            pass

    class _NoopPostprocessor:
        confidence_threshold, iou_threshold = 0.2, 0.45
        class_names = ["person"]

        def postprocess(self, outputs, image_shapes, restore_params=None):
            return [[] for _ in range(len(image_shapes))]

        def release(self):
            pass

    tmp = tempfile.mkdtemp()
    try:
        logger = init_logger({"log_path": os.path.join(tmp, "t.log")})
        model = DetectionModel(logger, _NoopEngine(), _NoopPreprocessor(),
                               _NoopPostprocessor(), model_type="yolox")
        # 合成5帧视频（fps=25）
        video_path = os.path.join(tmp, "v.mp4")
        writer = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*"mp4v"),
                                 25, (64, 64))
        for _ in range(5):
            writer.write(np.zeros((64, 64, 3), dtype=np.uint8))
        writer.release()
        # interval=0.02：0.02*25=0.5→int为0，修复前cnt%0崩溃
        result_path = os.path.join(tmp, "out.mp4")
        model.detect_video(video_path, result_path, interval=0.02)
        assert os.path.exists(result_path)
        cap = cv2.VideoCapture(result_path)
        frames = 0
        while cap.read()[0]:
            frames += 1
        cap.release()
        assert frames == 5, frames    # 无检测帧也写原帧，不丢帧
        model.release()
    finally:
        shutil.rmtree(tmp)


def test_eval_end_to_end_mAP_is_one():
    """GT与预测一致时COCO口径mAP必须≈1.0（验证gt坐标转换修复）"""
    tmp = tempfile.mkdtemp()
    try:
        # 合成VOC数据集（GT与FakeEngine输出完全一致）
        voc_dir = os.path.join(tmp, "VOC")
        _make_voc_dataset(voc_dir, n_images=2)
        classes_path = os.path.join(tmp, "classes.txt")
        with open(classes_path, "w") as f:
            f.write("person\ncar\n")
        model_path = os.path.join(tmp, "fake.onnx")
        open(model_path, "w").close()
        result_dir = os.path.join(tmp, "result")
        cfg = {"DetectionModel": {
            "model_type": "yolox", "engine_type": "fake",
            "engine_model_path": model_path, "class_name_path": classes_path,
            "confidence_threshold": 0.2, "iou_threshold": 0.45},
            "log_path": os.path.join(tmp, "t.log")}
        logger = init_logger(cfg)
        model = build_model(logger, cfg["DetectionModel"])

        from tools.eval import test as eval_test
        eval_test(logger, model, voc_dir, result_dir,
                  dataset_type="voc", choice="val", metrics="coco")

        # 1) 落盘JSON中gt的bbox必须是xywh（COCO口径），即[10,10,30,30]
        with open(os.path.join(result_dir, "gt_result.json")) as f:
            gt_json = json.load(f)
        gt_bbox = gt_json["annotations"][0]["bbox"]
        assert gt_bbox == [GT_BOX[0], GT_BOX[1],
                           GT_BOX[2] - GT_BOX[0], GT_BOX[3] - GT_BOX[1]], gt_bbox
        with open(os.path.join(result_dir, "detection_result.json")) as f:
            pred_json = json.load(f)
        pred_bbox = pred_json["annotations"][0]["bbox"]
        assert pred_bbox == [10, 10, 30, 30], pred_bbox

        # 2) COCOeval指标必须≈1.0
        _, mAP, mAP50, _ = eval_coco(
            os.path.join(result_dir, "gt_result.json"),
            os.path.join(result_dir, "detection_result.json"))
        print("mAP@0.5:0.95 = {:.4f}, mAP@0.5 = {:.4f}".format(mAP, mAP50))
        assert mAP50 > 0.99, "mAP@0.5={0} 应≈1.0（gt坐标转换若错误会显著偏低）".format(mAP50)
        assert mAP > 0.99, "mAP@0.5:0.95={0} 应≈1.0".format(mAP)
        model.release()
    finally:
        shutil.rmtree(tmp)


def test_eval_tail_batch_single_image():
    """回归：batch_size>1且尾批恰1张图时，eval.py曾按模型batch_size而非
    本批实际图像数判断解包裹——detect()对单图返回扁平列表，zip把图像与
    检测字典错配：有框时pred["bbox"]抛TypeError崩溃，无框时该图gt被静默
    丢弃致指标失真。3图→bs=2尾批1张，修复后3张gt全部落盘"""
    tmp = tempfile.mkdtemp()
    try:
        voc_dir = os.path.join(tmp, "VOC")
        _make_voc_dataset(voc_dir, n_images=3)      # 3图 → bs=2 尾批1张
        classes_path = os.path.join(tmp, "classes.txt")
        with open(classes_path, "w") as f:
            f.write("person\ncar\n")
        model_path = os.path.join(tmp, "fake.onnx")
        open(model_path, "w").close()
        result_dir = os.path.join(tmp, "result")
        cfg = {"DetectionModel": {
            "model_type": "yolox", "engine_type": "fake_batch2",
            "engine_model_path": model_path, "class_name_path": classes_path,
            "confidence_threshold": 0.2, "iou_threshold": 0.45},
            "log_path": os.path.join(tmp, "t.log")}
        logger = init_logger(cfg)
        model = build_model(logger, cfg["DetectionModel"])
        assert model.batch_size == 2, model.batch_size

        from tools.eval import test as eval_test
        eval_test(logger, model, voc_dir, result_dir,
                  dataset_type="voc", choice="val", metrics="coco")
        # 3张图的gt与图像信息全部落盘（尾批单图不被丢弃）
        with open(os.path.join(result_dir, "gt_result.json")) as f:
            gt_json = json.load(f)
        assert len(gt_json["annotations"]) == 3, len(gt_json["annotations"])
        assert len(gt_json["images"]) == 3, len(gt_json["images"])
        model.release()
    finally:
        shutil.rmtree(tmp)


def test_prelabel_merge_preserves_cross_class():
    """合并模式仅同类别才按IoU抑制（回归：曾只比坐标不比类别，不同类别
    空间重叠的对象预测框被误删，如"人"骑"自行车"）：预测bike与已有person
    框完全重叠但类别不同，合并后两者都应保留"""
    from tools.prelabel_labelme_dataset import detect_single_image

    class _Model:
        def __init__(self, preds):
            self.preds = preds

        def detect(self, image, export_time=False, print_detection_result=False):
            return self.preds

    class _Logger:
        def error(self, msg):
            pass

    tmp = tempfile.mkdtemp()
    try:
        # 已有json：person框[10,10,40,40]（4点矩形，labelme规范）
        img_path = os.path.join(tmp, "a.jpg")
        cv2.imwrite(img_path, np.zeros((64, 64, 3), dtype=np.uint8))
        json_path = os.path.join(tmp, "a.json")
        with open(json_path, "w") as f:
            json.dump({"version": "v0.0.1", "flags": {},
                       "shapes": [{"label": "person", "shape_type": "rectangle",
                                   "points": [[10, 10], [40, 10], [40, 40], [10, 40]]}],
                       "imagePath": "a.jpg", "imageData": None,
                       "imageHeight": 64, "imageWidth": 64}, f)
        # 预测bike框与person框完全重叠（同坐标不同类别）
        model = _Model([{"bbox": [10, 10, 40, 40], "cls_name": "bike"}])
        detect_single_image(_Logger(), model, img_path, json_path,
                            merge_existing_json=True, merge_iou_threshold=0.5)
        with open(json_path) as f:
            data = json.load(f)
        labels = sorted(s["label"] for s in data["shapes"])
        # 跨类别不抑制：person与bike都保留
        assert labels == ["bike", "person"], labels
    finally:
        shutil.rmtree(tmp)


def test_prelabel_systemic_error_reraised():
    """系统级错误（模型推理失败）原样重抛而非吞掉（回归：曾一并吞掉，
    整批白跑却退出码0的静默失败）；读图失败（单图级）仍隔离跳过"""
    from tools.prelabel_labelme_dataset import detect_single_image

    class _RaisingModel:
        def detect(self, image, export_time=False, print_detection_result=False):
            raise RuntimeError("engine inference failed")

    class _Logger:
        def __init__(self):
            self.errors = []

        def error(self, msg):
            self.errors.append(msg)

    tmp = tempfile.mkdtemp()
    try:
        img_path = os.path.join(tmp, "a.jpg")
        cv2.imwrite(img_path, np.zeros((64, 64, 3), dtype=np.uint8))
        json_path = os.path.join(tmp, "a.json")
        logger = _Logger()
        # 模型推理失败：RuntimeError 应 re-raise
        try:
            detect_single_image(logger, _RaisingModel(), img_path, json_path)
            assert False, "系统级错误应re-raise"
        except RuntimeError:
            pass
        assert logger.errors == [], logger.errors
        # 读图失败（不存在的图像）：单图级，隔离跳过并记error
        detect_single_image(logger, _RaisingModel(),
                            os.path.join(tmp, "nope.jpg"), json_path)
        assert len(logger.errors) == 1, logger.errors
    finally:
        shutil.rmtree(tmp)


def test_config_template():
    """配置模板结构可用（DetectionModel键齐全）"""
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cfg = load_yaml(os.path.join(repo_root, "config", "detection.yaml"))
    m = cfg["DetectionModel"]
    for key in ("model_type", "engine_type", "engine_model_path",
                "class_name_path", "confidence_threshold", "iou_threshold"):
        assert key in m, key


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
