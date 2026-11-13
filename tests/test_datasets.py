# -*- coding: utf-8 -*-
"""
    四种数据集加载器（VOC/COCO/yolo_v1/yolo_v2）的合成数据单元测试
    重点验证：gt内部统一x1y1x2y2语义、无标签负样本、多子集合并
"""

import os
import sys
import shutil
import tempfile

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from detection_track_toolkit.utils.dataset_utils.build import build_dataset

IMG = np.zeros((64, 64, 3), dtype=np.uint8)


def _make_image(path):
    cv2.imwrite(path, IMG)


def test_voc_dataset():
    root = tempfile.mkdtemp()
    try:
        os.makedirs(os.path.join(root, "Annotations"))
        os.makedirs(os.path.join(root, "JPEGImages"))
        os.makedirs(os.path.join(root, "ImageSets", "Main"))
        # 图1：1个目标；图2：无标签xml（负样本）
        for fid, has_obj in [("0001", True), ("0002", False)]:
            _make_image(os.path.join(root, "JPEGImages", fid + ".jpg"))
        with open(os.path.join(root, "Annotations", "0001.xml"), "w") as f:
            f.write('''<annotation><object><name>person</name><bndbox>
            <xmin>10</xmin><ymin>10</ymin><xmax>40</xmax><ymax>40</ymax>
            </bndbox></object></annotation>''')
        with open(os.path.join(root, "ImageSets", "Main", "val.txt"), "w") as f:
            f.write("0001\n0002\n")
        ds = build_dataset("voc", root, 1, "val")
        assert ds.get_image_num() == 2, ds.get_image_num()
        batch_imgs, meta = next(iter(ds))
        assert len(meta) == 1
        # gt内部表示：x1y1x2y2（三种数据集格式统一约定）
        obj = meta[0]["gt"][0]
        assert obj[0] == "person" and obj[1:5] == [10, 10, 40, 40], obj
    finally:
        shutil.rmtree(root)


def test_voc_manifest_with_extension_lines():
    """VOC清单行带后缀（0001.jpg）不再静默丢图（回归：曾只按id+后缀探测，
    带后缀行探测0001.jpg.jpg不存在被continue丢弃，数据集可静默变空）；
    且xml按去后缀id解析，带后缀行不被误判为无目标样本；空行跳过"""
    root = tempfile.mkdtemp()
    try:
        os.makedirs(os.path.join(root, "Annotations"))
        os.makedirs(os.path.join(root, "JPEGImages"))
        os.makedirs(os.path.join(root, "ImageSets", "Main"))
        for fid in ("0001", "0002", "0003"):
            _make_image(os.path.join(root, "JPEGImages", fid + ".jpg"))
            with open(os.path.join(root, "Annotations", fid + ".xml"), "w") as f:
                f.write('''<annotation><object><name>person</name><bndbox>
                <xmin>10</xmin><ymin>10</ymin><xmax>40</xmax><ymax>40</ymax>
                </bndbox></object></annotation>''')
        main_dir = os.path.join(root, "ImageSets", "Main")

        def load(manifest):
            with open(os.path.join(main_dir, "val.txt"), "w") as f:
                f.write(manifest)
            ds = build_dataset("voc", root, 1, "val")
            return ds

        # 三种清单写法均应加载全部3张（带后缀/裸id/混合，含空行）
        for manifest in ("0001\n0002\n0003\n",
                         "0001.jpg\n0002.jpg\n0003.jpg\n",
                         "0001\n\n0002.jpg\n0003\n"):
            ds = load(manifest)
            assert ds.get_image_num() == 3, (manifest, ds.get_image_num())
            # 带后缀行的xml正常解析（未被误判为无目标样本）
            for _, meta in ds:
                assert len(meta[0]["gt"]) == 1 and meta[0]["gt"][0][0] == "person", meta[0]
        # 图像目录中不存在的清单行：跳过但可从告警发现（数量不受影响地减少）
        ds = load("0001\n9999\n")
        assert ds.get_image_num() == 1, ds.get_image_num()
    finally:
        shutil.rmtree(root)


def test_get_classes_blank_line_handling():
    """类别名文件空行处理收口在get_classes（回归：yolo数据集曾无差别过滤
    所有空串，中部空行使id→类名映射前移导致静默错标）：
    末尾空行剔除（无害）、首部/中部空行报错并定位行号、空文件报错"""
    from detection_track_toolkit.detection_model.detection_utils import get_classes
    root = tempfile.mkdtemp()
    try:
        path = os.path.join(root, "classes.txt")
        # 末尾多个空行：剔除，id映射正确
        with open(path, "w") as f:
            f.write("person\ncar\nbus\n\n\n")
        names = get_classes(path)
        assert names == ["person", "car", "bus"], names
        # 首部空行：报错并定位行号
        with open(path, "w") as f:
            f.write("\nperson\ncar\n")
        try:
            get_classes(path)
            assert False, "首部空行应报错"
        except ValueError as e:
            assert "第1行" in str(e), e
        # 中部空行：报错并定位行号
        with open(path, "w") as f:
            f.write("person\ncar\n\nbus\n")
        try:
            get_classes(path)
            assert False, "中部空行应报错"
        except ValueError as e:
            assert "第3行" in str(e), e
        # 空文件（全空白行）：报错
        with open(path, "w") as f:
            f.write("\n \n")
        try:
            get_classes(path)
            assert False, "空文件应报错"
        except ValueError as e:
            assert "为空" in str(e), e
    finally:
        shutil.rmtree(root)


def test_yolo_v1_classes_file_blank_lines():
    """yolo_v1类别名文件：末尾空行正常加载且id→类名映射不变；
    中部空行加载即报错（不再静默错标后移）"""
    root = tempfile.mkdtemp()
    try:
        os.makedirs(os.path.join(root, "images"))
        os.makedirs(os.path.join(root, "labels"))
        _make_image(os.path.join(root, "images", "a.jpg"))
        # 标签类别id=1：末尾空行剔除后 class_names[1] 应仍为 'car'
        with open(os.path.join(root, "labels", "a.txt"), "w") as f:
            f.write("1 0.5 0.5 0.25 0.25\n")
        with open(os.path.join(root, "train.txt"), "w") as f:
            f.write("a.jpg\n")
        # 末尾空行：正常，gt类名为car（非错位后的第3类）
        with open(os.path.join(root, "classes.txt"), "w") as f:
            f.write("person\ncar\nbus\n\n")
        ds = build_dataset("yolo_v1", root, 1, "train")
        _, meta = next(iter(ds))
        assert meta[0]["gt"][0][0] == "car", meta[0]["gt"]
        # 中部空行：加载即报错
        with open(os.path.join(root, "classes.txt"), "w") as f:
            f.write("person\n\ncar\nbus\n")
        try:
            build_dataset("yolo_v1", root, 1, "train")
            assert False, "中部空行应报错"
        except ValueError as e:
            assert "第2行" in str(e), e
    finally:
        shutil.rmtree(root)


def test_yolo_label_blank_lines_and_malformed():
    """标签txt空行跳过（回归#6：曾对空行也走字段数校验报"行格式异常"，
    使整个数据集加载中止——空行是手编/拼接标签的常见形态）；
    非空但字段不足的畸形行仍显式报错"""
    root = tempfile.mkdtemp()
    try:
        os.makedirs(os.path.join(root, "images"))
        os.makedirs(os.path.join(root, "labels"))
        _make_image(os.path.join(root, "images", "a.jpg"))
        # 含中部/末尾空行的标签：应正常解析出2个目标
        with open(os.path.join(root, "labels", "a.txt"), "w") as f:
            f.write("0 0.5 0.5 0.25 0.25\n\n\n1 0.25 0.25 0.1 0.1\n")
        with open(os.path.join(root, "train.txt"), "w") as f:
            f.write("a.jpg\n")
        with open(os.path.join(root, "classes.txt"), "w") as f:
            f.write("person\ncar\n")
        ds = build_dataset("yolo_v1", root, 1, "train")
        _, meta = next(iter(ds))
        assert len(meta[0]["gt"]) == 2, meta[0]["gt"]
        # 非空畸形行（字段不足）仍报错
        with open(os.path.join(root, "labels", "a.txt"), "w") as f:
            f.write("0 0.5 0.5\n")
        try:
            build_dataset("yolo_v1", root, 1, "train")
            assert False, "字段不足应报错"
        except ValueError as e:
            assert "格式异常" in str(e), e
    finally:
        shutil.rmtree(root)


def test_dataset_image_ext_coverage():
    """图像后缀表覆盖bmp等格式（回归#8：曾仅jpg/jpeg/png六种变体，
    bmp/tif/webp图像被静默丢弃且无告警）：voc裸id清单探测bmp命中；
    yolo_v2目录遍历拾取bmp"""
    root = tempfile.mkdtemp()
    try:
        # voc：0001.bmp + 裸id清单（探测路径需覆盖.bmp）
        os.makedirs(os.path.join(root, "JPEGImages"))
        os.makedirs(os.path.join(root, "ImageSets", "Main"))
        _make_image(os.path.join(root, "JPEGImages", "0001.bmp"))
        with open(os.path.join(root, "ImageSets", "Main", "val.txt"), "w") as f:
            f.write("0001\n")
        ds = build_dataset("voc", root, 1, "val")
        assert ds.get_image_num() == 1, ds.get_image_num()
    finally:
        shutil.rmtree(root)
    root = tempfile.mkdtemp()
    try:
        # yolo_v2：val/images下为bmp图像（目录遍历需不跳过.bmp）
        os.makedirs(os.path.join(root, "val", "images"))
        os.makedirs(os.path.join(root, "val", "labels"))
        _make_image(os.path.join(root, "val", "images", "a.bmp"))
        with open(os.path.join(root, "val", "labels", "a.txt"), "w") as f:
            f.write("0 0.5 0.5 0.25 0.25\n")
        with open(os.path.join(root, "classes.txt"), "w") as f:
            f.write("person\n")
        ds = build_dataset("yolo_v2", root, 1, "val")
        assert ds.get_image_num() == 1, ds.get_image_num()
        _, meta = next(iter(ds))
        assert meta[0]["gt"][0][0] == "person", meta[0]["gt"]
    finally:
        shutil.rmtree(root)


def test_coco_dataset():
    root = tempfile.mkdtemp()
    try:
        os.makedirs(os.path.join(root, "val"))
        os.makedirs(os.path.join(root, "annotations"))
        _make_image(os.path.join(root, "val", "a.jpg"))
        ann = {
            "images": [{"id": 1, "file_name": "a.jpg", "width": 64, "height": 64}],
            "annotations": [{"image_id": 1, "category_id": 1,
                             "bbox": [10, 10, 30, 30], "id": 1}],
            "categories": [{"id": 1, "name": "person"}],
        }
        import json
        with open(os.path.join(root, "annotations", "val.json"), "w") as f:
            json.dump(ann, f)
        ds = build_dataset("coco", root, 1, "val")
        meta = ds.meta_data[0]
        obj = meta["gt"][0]
        # COCO标注bbox为xywh，加载器内部转x1y1x2y2（与VOC/YOLO对齐）
        assert obj[0] == "person" and obj[1:5] == [10, 10, 40, 40], obj
    finally:
        shutil.rmtree(root)


def test_yolo_v1_dataset():
    root = tempfile.mkdtemp()
    try:
        os.makedirs(os.path.join(root, "images"))
        os.makedirs(os.path.join(root, "labels"))
        _make_image(os.path.join(root, "images", "a.jpg"))
        # 64x64图，归一化cxcywh: 中心(25,25)=(0.390625,0.390625)，宽高30/64=0.46875
        # -> 像素x1y1x2y2 = (10,10,40,40)
        with open(os.path.join(root, "labels", "a.txt"), "w") as f:
            f.write("0 0.390625 0.390625 0.46875 0.46875\n")
        with open(os.path.join(root, "classes.txt"), "w") as f:
            f.write("person\ncar\n")
        with open(os.path.join(root, "val.txt"), "w") as f:
            f.write("a.jpg\n")
        ds = build_dataset("yolo_v1", root, 1, "val")
        meta = ds.meta_data[0]
        obj = meta["gt"][0]
        assert obj[0] == "person", obj
        assert np.allclose(obj[1:5], [10.0, 10.0, 40.0, 40.0], atol=1e-6), obj
        # 清单中无标签txt的图视为负样本
        _make_image(os.path.join(root, "images", "b.jpg"))
        with open(os.path.join(root, "val.txt"), "w") as f:
            f.write("a.jpg\nb.jpg\n")
        ds = build_dataset("yolo_v1", root, 1, "val")
        assert ds.get_image_num() == 2
        assert len(ds.meta_data[1]["gt"]) == 0
    finally:
        shutil.rmtree(root)


def test_yolo_v2_dataset():
    root = tempfile.mkdtemp()
    try:
        os.makedirs(os.path.join(root, "val", "images"))
        os.makedirs(os.path.join(root, "val", "labels"))
        _make_image(os.path.join(root, "val", "images", "a.jpg"))
        # 64x64图，归一化cxcywh: 中心(25,25)=(0.390625,0.390625)，宽高30/64=0.46875
        # -> 像素x1y1x2y2 = (10,10,40,40)
        with open(os.path.join(root, "val", "labels", "a.txt"), "w") as f:
            f.write("0 0.390625 0.390625 0.46875 0.46875\n")
        with open(os.path.join(root, "classes.txt"), "w") as f:
            f.write("person\ncar\n")
        ds = build_dataset("yolo_v2", root, 1, "val")
        meta = ds.meta_data[0]
        obj = meta["gt"][0]
        assert obj[0] == "person", obj
        assert np.allclose(obj[1:5], [10.0, 10.0, 40.0, 40.0], atol=1e-6), obj
        # 无标签txt的图视为负样本
        _make_image(os.path.join(root, "val", "images", "b.jpg"))
        ds = build_dataset("yolo_v2", root, 1, "val")
        assert ds.get_image_num() == 2
        assert len(ds.meta_data[1]["gt"]) == 0
    finally:
        shutil.rmtree(root)


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
            print("FAIL", name, "->", type(e).__name__, str(e)[:160])
    print("=" * 40, "fails:", fails)
    return fails


if __name__ == "__main__":
    sys.exit(1 if _run_all() else 0)
