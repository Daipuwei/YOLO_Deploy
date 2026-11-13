# -*- coding: utf-8 -*-
"""
    后处理纯函数与各风格后处理器解码的单元测试（pytest风格，可python直接运行）
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from detection_track_toolkit.preprocess.preprocess_utils import resize_keep_aspect_ratio
from detection_track_toolkit.preprocess.preprocess_utils import resize
from detection_track_toolkit.postprocess.postprocess_utils import dist2bbox
from detection_track_toolkit.postprocess.postprocess_utils import iou_xyxy, nms, rescale_coords
from detection_track_toolkit.postprocess import build_postprocessor
from detection_track_toolkit.postprocess import POSTPROCESSOR_REGISTRY

CLASS_NAMES = ["person", "car"]


def test_registry_complete():
    """三个风格后处理器均已注册"""
    assert sorted(POSTPROCESSOR_REGISTRY.keys()) == ["dflyolo_postprocess", "yolov5_postprocess", "yolox_postprocess"]


def test_iou_xyxy():
    box = np.array([0, 0, 10, 10], dtype=np.float32)
    boxes = np.array([[5, 5, 15, 15], [20, 20, 30, 30], [0, 0, 10, 10]], dtype=np.float32)
    iou = iou_xyxy(box, boxes, "iou")
    assert np.isclose(iou[0], 25.0 / 175.0), iou          # 相交25，并集175
    assert np.isclose(iou[1], 0.0), iou                   # 不相交
    assert np.isclose(iou[2], 1.0), iou                   # 完全重叠
    giou = iou_xyxy(box, boxes, "giou")
    assert np.all(giou <= iou + 1e-6), giou               # giou<=iou
    diou = iou_xyxy(box, boxes, "diou")
    assert np.all(diou <= iou + 1e-6), diou               # diou<=iou


def test_nms_class_agnostic():
    preds = np.array([[0, 0, 10, 10, 0.9, 0],
                      [1, 1, 11, 11, 0.8, 0],      # 与第一个高重叠，应被抑制
                      [20, 20, 30, 30, 0.7, 1],    # 独立框，保留
                      [0, 0, 2, 2, 0.1, 0]], dtype=np.float32)  # 低分，过滤
    keep = nms(preds, 0.2, 0.45, mode="class_agnostic")
    assert sorted(keep.tolist()) == [0, 2], keep


def test_nms_per_class():
    """per_class模式：不同类别中度重叠(IoU~0.54)的框互不抑制；
    完全重叠(IoU=1)视为同一目标的多类别重复预测，第二阶段全局过滤
    缺省（None）必须执行、阈值取iou_threshold抑制；显式0仍可禁用"""
    # 中度重叠：IoU=70/130≈0.538，介于0.45与0.9之间
    preds = np.array([[0, 0, 10, 10, 0.9, 0],
                      [3, 0, 13, 10, 0.8, 1]], dtype=np.float32)
    keep_agnostic = nms(preds, 0.2, 0.45, mode="class_agnostic")
    assert sorted(keep_agnostic.tolist()) == [0], keep_agnostic  # 类别无关：0.538>0.45抑制
    keep_per_class = nms(preds, 0.2, 0.45, mode="per_class",
                         class_agnostic_threshold=0.9)
    assert sorted(keep_per_class.tolist()) == [0, 1], keep_per_class  # 0.538<0.9两阶段保留
    # 完全重叠：第二阶段全局过滤抑制跨类别重复框（同一目标被多类别预测）
    preds_full = np.array([[0, 0, 10, 10, 0.9, 0],
                           [0, 0, 10, 10, 0.8, 1]], dtype=np.float32)
    keep_full = nms(preds_full, 0.2, 0.45, mode="per_class",
                    class_agnostic_threshold=0.9)
    assert len(keep_full) == 1, keep_full
    # 缺省None：第二阶段必须执行，完全重叠的跨类别框按iou_threshold抑制
    keep_none = nms(preds_full, 0.2, 0.45, mode="per_class")
    assert len(keep_none) == 1, keep_none
    # 显式0：禁用全局过滤，完全重叠的跨类别框全部保留
    keep_zero = nms(preds_full, 0.2, 0.45, mode="per_class",
                    class_agnostic_threshold=0)
    assert sorted(keep_zero.tolist()) == [0, 1], keep_zero


def test_rescale_coords():
    """keep_aspect_ratio gain还原：模型320x320、原图100x200，gain=1.6"""
    preds = np.array([[16, 32, 64, 96, 0.9, 0]], dtype=np.float32)
    out = rescale_coords(preds, (320, 320), (100, 200))
    assert np.allclose(out[0, 0:4], [10, 20, 40, 60], atol=1e-3), out


def test_letterbox_params_pad_modes():
    """letterbox填充参数两种锚定：top_left全填右下；center上下均分
    （100x200→320x320：缩放到160x320，dh=160,dw=0）"""
    from detection_track_toolkit.preprocess.preprocess_utils import letterbox_params
    # top_left：贴左上角，填充全在bottom
    new_h, new_w, top, bottom, left, right = letterbox_params((100, 200), (320, 320))
    assert (new_h, new_w) == (160, 320)
    assert (top, left) == (0, 0) and (bottom, right) == (160, 0)
    # center：垂直方向均分80/80，水平无填充
    new_h, new_w, top, bottom, left, right = letterbox_params(
        (100, 200), (320, 320), pad_mode="center")
    assert (new_h, new_w) == (160, 320)
    assert top + bottom == 160 and abs(top - bottom) <= 1
    assert (left, right) == (0, 0)
    # 非法模式报错
    try:
        letterbox_params((100, 200), (320, 320), pad_mode="middle")
        assert False, "非法pad_mode应报错"
    except ValueError:
        pass


def test_letterbox_params_extreme_aspect_ratio():
    """极端宽高比下缩放后尺寸下限钳到1（回归：1×1280/2×2560条带时
    h*r<0.5，round得new_h=0，使cv2.resize(dsize含0)崩溃且gain_h除零；
    钳到1后new_h>=1且gain非零可安全还原）"""
    from detection_track_toolkit.preprocess.preprocess_utils import letterbox_params
    for image_hw, expected_min_dim in (((1, 1280), 640),   # 极宽条带：h→1
                                       ((2, 2560), 640),
                                       ((50, 64000), 640),
                                       ((1280, 1), 640),   # 极瘦条带：w→1
                                       ((2560, 2), 640)):
        new_h, new_w, top, bottom, left, right = letterbox_params(image_hw, (640, 640))
        assert new_h >= 1 and new_w >= 1, (image_hw, new_h, new_w)
        assert new_h + top + bottom == 640, (image_hw, new_h, top, bottom)
        assert new_w + left + right == 640, (image_hw, new_w, left, right)


def test_rescale_coords_with_restore_params():
    """restore_params精确逆变换还原（修复#9：拉伸模式单一gain曾错1.78倍、
    居中填充曾无偏移还原）：
    - 居中letterbox（100x200→320x320，gain=1.6，pad_h=80）：需减偏移
    - resize拉伸（100x200→320x320，gain_h=3.2/gain_w=1.6）：双轴独立
    - None缺省走旧口径（单一gain，top_left等价）"""
    # 居中letterbox：模型坐标y含80px填充偏移
    preds = np.array([[16, 96, 64, 128, 0.9, 0]], dtype=np.float32)
    out = rescale_coords(preds, (320, 320), (100, 200),
                         restore_params=(1.6, 1.6, 80.0, 0.0))
    assert np.allclose(out[0, 0:4], [10, 10, 40, 30], atol=1e-3), out
    # 拉伸模式：x/y各除各的gain（x: /1.6，y: /3.2）
    preds = np.array([[16, 32, 64, 96, 0.9, 0]], dtype=np.float32)
    out = rescale_coords(preds, (320, 320), (100, 200),
                         restore_params=(3.2, 1.6, 0.0, 0.0))
    assert np.allclose(out[0, 0:4], [10, 10, 40, 30], atol=1e-3), out
    # None缺省：旧口径（单一min gain，等价top_left letterbox）
    preds = np.array([[16, 32, 64, 96, 0.9, 0]], dtype=np.float32)
    out = rescale_coords(preds, (320, 320), (100, 200))
    assert np.allclose(out[0, 0:4], [10, 20, 40, 60], atol=1e-3), out


def test_compute_restore_params_roundtrip():
    """compute_restore_params与resize_image互为精确逆变换（两种模式两种锚定）：
    模型坐标正变换→restore_params还原→还原回原图坐标"""
    from detection_track_toolkit.preprocess.base_preprocessor import BasePreprocessor

    class _ProbePreprocessor(BasePreprocessor):
        """最小具体子类（实现抽象方法即可，测试仅用基类共享辅助）"""
        def preprocess_single(self, image):
            return None, None, None

    image = np.random.randint(0, 255, (100, 200, 3), dtype=np.uint8)
    for resize_mode, pad_mode in (("keep_aspect_ratio", "top_left"),
                                  ("keep_aspect_ratio", "center"),
                                  ("resize", "top_left")):
        pre = _ProbePreprocessor((1, 3, 320, 320), is_nchw=True,
                                 resize_mode=resize_mode, pad_mode=pad_mode)
        resized = pre.resize_image(image)
        assert resized.shape[0:2] == (320, 320)
        restore = pre.compute_restore_params(100, 200)
        # 原图四个角点正变换到模型坐标，再经restore_params还原
        corners = np.array([[0, 0], [199, 0], [0, 99], [199, 99]], dtype=np.float32)
        model_xy = np.stack([corners[:, 0] * restore[1] + restore[3],
                             corners[:, 1] * restore[0] + restore[2]], axis=1)
        preds = np.concatenate([model_xy, np.zeros((4, 2), dtype=np.float32)], axis=1)
        restored = rescale_coords(preds, (320, 320), (100, 200),
                                  restore_params=restore)
        assert np.allclose(restored[:, 0:2], corners, atol=1.0), \
            (resize_mode, pad_mode, restored[:, 0:2])


def test_resize_keep_aspect_ratio_roundtrip():
    """resize_keep_aspect_ratio（原letterbox）只做右下填充，与scale_coords只除gain的约定自洽"""
    img = np.zeros((100, 200, 3), dtype=np.uint8)
    padded = resize_keep_aspect_ratio(img, (320, 320))
    assert padded.shape == (320, 320, 3)
    # 尺寸一致快速路径：直接返回原图（零拷贝，同一对象）
    img2 = np.zeros((320, 320, 3), dtype=np.uint8)
    assert resize_keep_aspect_ratio(img2, (320, 320)) is img2


def test_resize():
    """resize直接拉伸填满目标尺寸，尺寸一致时零拷贝直返"""
    img = np.zeros((100, 200, 3), dtype=np.uint8)
    filled = resize(img, (320, 320))
    assert filled.shape == (320, 320, 3)
    # 尺寸一致快速路径：直接返回原图（零拷贝，同一对象）
    img2 = np.zeros((320, 320, 3), dtype=np.uint8)
    assert resize(img2, (320, 320)) is img2


def test_yolov5_decode():
    pp = build_postprocessor("yolov5_postprocess", 0.2, 0.45, CLASS_NAMES, (320, 320))
    # 融合输出(N,5+nc): [cx,cy,w,h,obj,cls0,cls1]
    dets = np.array([[20, 30, 20, 20, 0.9, 0.95, 0.1],   # 框(10,20,30,40)类0
                     [100, 100, 10, 10, 0.9, 0.1, 0.95], # 框(95,95,105,105)类1
                     [200, 200, 10, 10, 0.1, 0.99, 0.9], # obj过低应被预过滤
                     ], dtype=np.float32)
    preds = pp.decode(dets)
    assert preds.shape[0] == 2, preds
    assert np.allclose(preds[0, 0:4], [10, 20, 30, 40]), preds
    assert np.isclose(preds[0, 4], 0.95 * 0.9), preds     # score=obj*cls
    assert preds[0, 5] == 0 and preds[1, 5] == 1, preds
    # 两阶段过滤：obj高但类别最大分低→滤除（旧逻辑按score=obj*cls会漏出）
    dets2 = np.array([[20, 30, 20, 20, 0.9, 0.05, 0.1]], dtype=np.float32)
    preds2 = pp.decode(dets2)
    assert preds2.shape[0] == 0, preds2
    # obj阶段全滤除：返回空预测数组（不进入argmax，避免空数组崩溃）
    dets3 = np.array([[20, 30, 20, 20, 0.05, 0.99, 0.9]], dtype=np.float32)
    preds3 = pp.decode(dets3)
    assert preds3.shape[0] == 0, preds3
    # 完整后处理（含尺度还原与NMS）
    outs = pp.postprocess_single(dets, (100, 200))
    assert len(outs) == 2 and outs[0]["cls_name"] == "person", outs


def test_dflyolo_decode_variants():
    """单输出ONNX格式确认：4+1+num_classes（scores_sum在第4列，可缺省）"""
    pp = build_postprocessor("dflyolo_postprocess", 0.2, 0.45, CLASS_NAMES, (320, 320))
    # 4+nc（无scores_sum）: [x1,y1,x2,y2,cls0,cls1]
    dets = np.array([[10, 20, 30, 40, 0.95, 0.1],
                     [5, 5, 8, 8, 0.1, 0.1]], dtype=np.float32)
    preds = pp.decode(dets)
    assert preds.shape[0] == 1 and preds[0, 5] == 0, preds
    # 5+nc（scores_sum在第4列）: [x1,y1,x2,y2,ss,cls0,cls1]
    dets5 = np.array([[10, 20, 30, 40, 0.9, 0.1, 0.3],   # scores_sum高→保留，cls1分高
                      [5, 5, 8, 8, 0.05, 0.9, 0.1]], dtype=np.float32)
    preds5 = pp.decode(dets5)
    assert preds5.shape[0] == 1 and preds5[0, 5] == 1, preds5
    # scores_sum存在时两阶段过滤：ss高但类别最大分低→滤除；ss与类分均高→保留
    dets5b = np.array([[10, 20, 30, 40, 0.9, 0.05, 0.1],   # ss=0.9高、cls最大分0.1低→滤除
                       [50, 50, 60, 60, 0.9, 0.6, 0.1]], dtype=np.float32)
    preds5b = pp.decode(dets5b)
    assert preds5b.shape[0] == 1, preds5b
    assert preds5b[0, 5] == 0 and np.allclose(preds5b[0, 0:4], [50, 50, 60, 60]), preds5b
    # scores_sum阶段全滤除：返回空预测数组（不进入argmax，避免空数组崩溃）
    dets5c = np.array([[10, 20, 30, 40, 0.05, 0.9, 0.1]], dtype=np.float32)
    preds5c = pp.decode(dets5c)
    assert preds5c.shape[0] == 0, preds5c
    # 列数非法：明确报错
    try:
        pp.decode(np.zeros((1, 9), dtype=np.float32))
        raise AssertionError("非法列数未被拦截")
    except ValueError as e:
        assert "不符" in str(e)
    # dim_first布局（导出带transpose）：(output_dim, grid_num)输入
    pp_t = build_postprocessor("dflyolo_postprocess", 0.2, 0.45, CLASS_NAMES, (320, 320),
                               output_layout="dim_first")
    # (6,1)：单anchor，每行一个分量的值
    dets_t = np.array([[10], [20], [30], [40], [0.95], [0.1]], dtype=np.float32)
    preds_t = pp_t.decode(dets_t)
    assert preds_t.shape[0] == 1 and preds_t[0, 5] == 0, preds_t
    assert np.allclose(preds_t[0, 0:4], [10, 20, 30, 40]), preds_t
    # cxcywh布局
    pp_cx = build_postprocessor("dflyolo_postprocess", 0.2, 0.45, CLASS_NAMES, (320, 320),
                                bbox_format="cxcywh")
    dets_cx = np.array([[20, 30, 20, 20, 0.95, 0.1]], dtype=np.float32)
    preds_cx = pp_cx.decode(dets_cx)
    assert np.allclose(preds_cx[0, 0:4], [10, 20, 30, 40]), preds_cx


def test_yolox_decode():
    pp = build_postprocessor("yolox_postprocess", 0.2, 0.45, CLASS_NAMES, (320, 320))
    dets = np.array([[10, 20, 30, 40, 0.9, 0.8, 0.1]], dtype=np.float32)
    preds = pp.decode(dets)
    assert np.isclose(preds[0, 4], 0.8 * 0.9), preds     # score=cls_max*obj
    assert preds[0, 5] == 0, preds
    # 两阶段过滤：obj高但类别最大分低→滤除；obj与类分均高→保留；obj低→滤除
    dets2 = np.array([[10, 20, 30, 40, 0.9, 0.05, 0.1],   # obj=0.9高、cls最大分0.1低→滤除
                      [50, 50, 60, 60, 0.9, 0.6, 0.1],    # obj与cls均高→保留
                      [70, 70, 80, 80, 0.05, 0.9, 0.1]], dtype=np.float32)
    preds2 = pp.decode(dets2)
    assert preds2.shape[0] == 1, preds2
    assert preds2[0, 5] == 0 and np.allclose(preds2[0, 0:4], [50, 50, 60, 60]), preds2
    assert np.isclose(preds2[0, 4], 0.6 * 0.9), preds2    # score=cls_max*obj
    # obj阶段全滤除：返回空预测数组（不进入argmax，避免空数组崩溃）
    dets3 = np.array([[10, 20, 30, 40, 0.05, 0.9, 0.1]], dtype=np.float32)
    preds3 = pp.decode(dets3)
    assert preds3.shape[0] == 0, preds3
    # cxcywh布局变体
    pp_cx = build_postprocessor("yolox_postprocess", 0.2, 0.45, CLASS_NAMES, (320, 320),
                                bbox_format="cxcywh")
    dets_cx = np.array([[20, 30, 20, 20, 0.9, 0.8, 0.1]], dtype=np.float32)
    preds_cx = pp_cx.decode(dets_cx)
    assert np.allclose(preds_cx[0, 0:4], [10, 20, 30, 40]), preds_cx
    assert np.isclose(preds_cx[0, 4], 0.8 * 0.9), preds_cx
    # dim_first布局（导出带transpose）：(output_dim, grid_num)输入
    pp_t = build_postprocessor("yolox_postprocess", 0.2, 0.45, CLASS_NAMES, (320, 320),
                               output_layout="dim_first")
    # (7,2)：两行分别为两个anchor，每列为一个分量的7个值
    dets_t = np.array([[10, 5], [20, 5], [30, 8], [40, 8],
                       [0.9, 0.1], [0.8, 0.1], [0.1, 0.1]], dtype=np.float32)
    preds_t = pp_t.decode(dets_t)
    assert np.allclose(preds_t[0, 0:4], [10, 20, 30, 40]), preds_t
    assert np.isclose(preds_t[0, 4], 0.8 * 0.9), preds_t
    # 列数非法：明确报错
    try:
        pp.decode(np.zeros((1, 8), dtype=np.float32))
        raise AssertionError("非法列数未被拦截")
    except ValueError as e:
        assert "不符" in str(e)


def test_yolov5_multioutputs_decode():
    """yolov5多输出解码（类方法multioutputs_decode，参考yolox流程）：
    grid/anchor解码为[cx,cy,w,h,obj,cls...]，复用现有decode"""
    num_classes = len(CLASS_NAMES)
    anchors = [[[10, 13], [16, 30], [33, 23]]]
    model_hw = (64, 64)
    h, w = 8, 8
    branch = np.zeros((1, 3 * (5 + num_classes), h, w), dtype=np.float32)
    # 层0首个anchor、首个grid：x=y=w=h=0.5、obj=1、cls0=0.9
    # -> cx=(0.5*2-0.5+0)*8=4、cy=4、bw=(0.5*2)^2*10=10、bh=13
    branch[0, 0, 0, 0] = 0.5   # x
    branch[0, 1, 0, 0] = 0.5   # y
    branch[0, 2, 0, 0] = 0.5   # w
    branch[0, 3, 0, 0] = 0.5   # h
    branch[0, 4, 0, 0] = 1.0   # obj
    branch[0, 5, 0, 0] = 0.9   # cls0
    pp = build_postprocessor("yolov5_postprocess", 0.2, 0.45, CLASS_NAMES, model_hw,
                             anchors=anchors, with_sigmoid=True)
    fused = pp.multioutputs_decode([branch])
    assert len(fused) == 1 and fused[0].shape == (3 * h * w, 5 + num_classes), fused[0].shape
    assert np.allclose(fused[0][0, 0:4], [4, 4, 10, 13]), fused[0][0]
    assert np.isclose(fused[0][0, 4], 1.0) and np.isclose(fused[0][0, 5], 0.9), fused[0][0]
    # 融合数组直接复用现有后处理器decode（cxcywh->xyxy）
    preds = pp.decode(fused[0])
    assert np.allclose(preds[0, 0:4], [-1.0, -2.5, 9.0, 10.5]), preds
    assert np.isclose(preds[0, 4], 0.9) and preds[0, 5] == 0, preds
    # with_sigmoid=False：x/y/w/h/obj/cls logit=0 -> sigmoid后0.5，坐标与上一态一致
    branch_logit = np.zeros_like(branch)
    pp_logit = build_postprocessor("yolov5_postprocess", 0.2, 0.45, CLASS_NAMES, model_hw,
                                   anchors=anchors, with_sigmoid=False)
    fused_logit = pp_logit.multioutputs_decode([branch_logit])
    assert np.allclose(fused_logit[0][0, 0:4], [4, 4, 10, 13]), fused_logit[0][0]
    assert np.isclose(fused_logit[0][0, 4], 0.5) and np.isclose(fused_logit[0][0, 5], 0.5), fused_logit[0][0]
    # 2D布局(C,h*w)与3D布局(C,h,w)融合结果一致（stride经sqrt自动推导）
    fused_2d = pp.multioutputs_decode([branch.reshape(1, branch.shape[1], -1)])
    assert np.allclose(fused_2d[0], fused[0]), fused_2d[0]
    # batch=2：逐图像独立融合
    fused_b2 = pp.multioutputs_decode([np.concatenate([branch, branch], axis=0)])
    assert len(fused_b2) == 2 and np.allclose(fused_b2[0], fused[0]) \
        and np.allclose(fused_b2[1], fused[0]), fused_b2
    # 缺anchors：显式报错
    pp_bare = build_postprocessor("yolov5_postprocess", 0.2, 0.45, CLASS_NAMES, model_hw)
    try:
        pp_bare.multioutputs_decode([branch])
        raise AssertionError("缺anchors未被拦截")
    except ValueError as e:
        assert "anchors" in str(e)
    # 输出通道数非法：显式报错
    try:
        pp.multioutputs_decode([np.zeros((1, 20, h, w), dtype=np.float32)])
        raise AssertionError("非法通道数未被拦截")
    except ValueError as e:
        assert "不符" in str(e)
    # 3输出模型（stride 8/16/32，由model_hw与特征图hw自动推导）：
    # reshape为(bs,output_dim,H*W/64|256|1024)后按grid维concat，
    # 逐网格stride数组与逐层anchor解码（anchor-major行序）
    anchors3 = [[[10, 13], [16, 30], [33, 23]],
                [[30, 61], [62, 45], [59, 119]],
                [[116, 90], [156, 198], [373, 326]]]
    b0 = np.zeros((1, 3 * (5 + num_classes), 8, 8), dtype=np.float32)   # H*W/64=64
    b1 = np.zeros((1, 3 * (5 + num_classes), 4, 4), dtype=np.float32)   # H*W/256=16
    b2 = np.zeros((1, 3 * (5 + num_classes), 2, 2), dtype=np.float32)   # H*W/1024=4
    b0[0, 0:4, 0, 0] = 0.5   # 层0 anchor0 grid(0,0)：x=y=w=h=0.5
    b0[0, 7:11, 0, 0] = 0.5  # 层0 anchor1 grid(0,0)
    b0[0, 14:18, 0, 0] = 0.5  # 层0 anchor2 grid(0,0)
    b1[0, 0:4, 0, 0] = 0.5   # 层1 anchor0 grid(0,0)
    b2[0, 0:4, 0, 0] = 0.5   # 层2 anchor0 grid(0,0)
    pp3 = build_postprocessor("yolov5_postprocess", 0.2, 0.45, CLASS_NAMES, model_hw,
                              anchors=anchors3)
    fused3 = pp3.multioutputs_decode([b0, b1, b2])
    assert fused3[0].shape == (3 * (64 + 16 + 4), 5 + num_classes), fused3[0].shape
    # 行0/64/128：层0 anchor0/1/2首个grid（cxcywh格式）：
    # cx=cy=(0.5*2-0.5+0)*8=4，bw/bh=anchor宽高
    assert np.allclose(fused3[0][0, 0:4], [4, 4, 10, 13]), fused3[0][0]
    assert np.allclose(fused3[0][64, 0:4], [4, 4, 16, 30]), fused3[0][64]
    assert np.allclose(fused3[0][128, 0:4], [4, 4, 33, 23]), fused3[0][128]
    # 行192/240：层1/层2 anchor0首个grid（stride 16/32）：
    # cx=cy=(0.5*2-0.5+0)*stride，bw/bh=各层anchor0宽高
    assert np.allclose(fused3[0][192, 0:4], [8, 8, 30, 61]), fused3[0][192]
    assert np.allclose(fused3[0][240, 0:4], [16, 16, 116, 90]), fused3[0][240]
    # 2D布局分支：结果与3D一致
    fused3_2d = pp3.multioutputs_decode([b0.reshape(1, 3 * (5 + num_classes), -1),
                                         b1.reshape(1, 3 * (5 + num_classes), -1),
                                         b2.reshape(1, 3 * (5 + num_classes), -1)])
    assert np.allclose(fused3_2d[0], fused3[0]), fused3_2d[0]
    # 类内调用路径：postprocess遇多输出列表自动经multioutputs_decode融合，
    # 再走单输出管线（尺度还原->NMS->编码）
    b0[0, 4, 0, 0] = 1.0    # obj=1
    b0[0, 5, 0, 0] = 0.9    # cls0=0.9
    outs = pp3.postprocess([b0, b1, b2], [(64, 64)])
    assert len(outs[0]) == 1, outs
    assert outs[0][0]["cls_name"] == "person", outs
    assert np.isclose(outs[0][0]["score"], 0.9), outs


def test_yolox_multioutputs_decode():
    """yolox多输出解码（类方法multioutputs_decode）：grid/stride解码为
    [x1,y1,x2,y2,obj,cls...]，复用现有decode/postprocess管线"""
    num_classes = len(CLASS_NAMES)
    model_hw = (64, 64)
    h, w = 8, 8
    branch = np.zeros((1, 5 + num_classes, h, w), dtype=np.float32)
    # 首个grid：x=y=0.5（已激活）、w=h=0（wh=exp(0)*8=8）、obj=1、cls0=0.9
    # -> xy=0.5*8+0*8=4 -> box [0,0,8,8]
    branch[0, 0, 0, 0] = 0.5   # x
    branch[0, 1, 0, 0] = 0.5   # y
    branch[0, 4, 0, 0] = 1.0   # obj
    branch[0, 5, 0, 0] = 0.9   # cls0
    pp = build_postprocessor("yolox_postprocess", 0.2, 0.45, CLASS_NAMES, model_hw,
                             with_sigmoid=True)
    fused = pp.multioutputs_decode([branch])
    assert len(fused) == 1 and fused[0].shape == (h * w, 5 + num_classes), fused[0].shape
    # 融合恒输出cxcywh（中心点=(0.5*8+0, 0.5*8+0)=(4,4)，wh=(8,8)），
    # 且output_layout被重新赋值为grid_first
    assert np.allclose(fused[0][0, 0:4], [4, 4, 8, 8]), fused[0][0]
    assert pp.output_layout == "grid_first" and pp.bbox_format == "cxcywh"
    preds = pp.decode(fused[0])
    assert np.allclose(preds[0, 0:4], [0, 0, 8, 8]), preds     # decode完成cxcywh->xyxy
    assert np.isclose(preds[0, 4], 0.9) and preds[0, 5] == 0, preds
    # with_sigmoid=False：bbox不做sigmoid（x=y=0 -> 中心点=(0,0)，
    # wh=exp(0)*8=8 -> cxcywh=[0,0,8,8]）；obj/cls logit=0 -> sigmoid后0.5
    branch_logit = np.zeros_like(branch)
    pp_logit = build_postprocessor("yolox_postprocess", 0.2, 0.45, CLASS_NAMES, model_hw,
                                   with_sigmoid=False)
    fused_logit = pp_logit.multioutputs_decode([branch_logit])
    assert np.allclose(fused_logit[0][0, 0:4], [0, 0, 8, 8]), fused_logit[0][0]
    assert np.isclose(fused_logit[0][0, 4], 0.5) and np.isclose(fused_logit[0][0, 5], 0.5), fused_logit[0][0]
    # 不同bbox_format配置实例：融合输出恒为cxcywh（配置不影响融合格式）
    pp_cx = build_postprocessor("yolox_postprocess", 0.2, 0.45, CLASS_NAMES, model_hw,
                                bbox_format="cxcywh")
    fused_cx = pp_cx.multioutputs_decode([branch])
    assert np.allclose(fused_cx[0][0, 0:4], [4, 4, 8, 8]), fused_cx[0][0]
    # 输出通道数非法：显式报错
    try:
        pp.multioutputs_decode([np.zeros((1, 6, h, w), dtype=np.float32)])
        raise AssertionError("非法通道数未被拦截")
    except ValueError as e:
        assert "不符" in str(e)
    # 3输出模型（stride 8/16/32，由model_hw与特征图hw自动推导）：
    # reshape为(bs,output_dim,H*W/64|256|1024)后按grid维concat，
    # 逐网格stride数组解码（中心点=x*stride+偏置，wh=exp*stride）
    model_hw3 = (64, 64)
    b0 = np.zeros((1, 5 + num_classes, 8, 8), dtype=np.float32)     # H*W/64=64
    b1 = np.zeros((1, 5 + num_classes, 4, 4), dtype=np.float32)     # H*W/256=16
    b2 = np.zeros((1, 5 + num_classes, 2, 2), dtype=np.float32)     # H*W/1024=4
    # 各层grid(0,0)：x=y=0.5（已激活）、w=h=0（wh=exp(0)*stride）
    b0[0, 0, 0, 0] = b0[0, 1, 0, 0] = 0.5
    b1[0, 0, 0, 0] = b1[0, 1, 0, 0] = 0.5
    b2[0, 0, 0, 0] = b2[0, 1, 0, 0] = 0.5
    pp3 = build_postprocessor("yolox_postprocess", 0.2, 0.45, CLASS_NAMES, model_hw3)
    fused3 = pp3.multioutputs_decode([b0, b1, b2])
    assert fused3[0].shape == (64 + 16 + 4, 5 + num_classes), fused3[0].shape
    # 行0/64/80分别为三层的首个grid（cxcywh格式）：中心点=(0.5+0)*stride，wh=stride
    assert np.allclose(fused3[0][0, 0:4], [4, 4, 8, 8]), fused3[0][0]
    assert np.allclose(fused3[0][64, 0:4], [8, 8, 16, 16]), fused3[0][64]
    assert np.allclose(fused3[0][80, 0:4], [16, 16, 32, 32]), fused3[0][80]
    # 非首网格偏置验证：层0 grid(1,0)（x=1,y=0）：中心点=(0.5*8+1*8, 0.5*8+0*8)=(12,4)
    b0[0, 0, 0, 1] = b0[0, 1, 0, 1] = 0.5
    fused3b = pp3.multioutputs_decode([b0, b1, b2])
    assert np.allclose(fused3b[0][1, 0:4], [12, 4, 8, 8]), fused3b[0][1]
    # 2D布局分支（导出带reshape/transpose）：grid经model_hw/stride推导，结果一致
    fused3_2d = pp3.multioutputs_decode([b0.reshape(1, 5 + num_classes, -1),
                                         b1.reshape(1, 5 + num_classes, -1),
                                         b2.reshape(1, 5 + num_classes, -1)])
    assert np.allclose(fused3_2d[0], fused3b[0]), fused3_2d[0]
    # 类内调用路径：postprocess遇多输出列表自动经multioutputs_decode融合，
    # 再走单输出管线（尺度还原->NMS->编码）
    b0[0, 4, 0, 0] = 1.0      # obj=1
    b0[0, 5, 0, 0] = 0.9      # cls0=0.9
    outs = pp3.postprocess([b0, b1, b2], [(64, 64)])
    assert len(outs[0]) == 1, outs
    assert outs[0][0]["cls_name"] == "person", outs
    assert np.isclose(outs[0][0]["score"], 0.9), outs


def test_dflyolo_multioutputs_decode():
    """DFL多输出解码（类方法multioutputs_decode）：DFL回归+dist2bbox解码为
    [x1,y1,x2,y2,(scores_sum),cls...]，复用现有decode/postprocess管线"""
    num_classes = len(CLASS_NAMES)
    reg_max = 2     # 仅用于构造测试分支的通道数，实例reg_max由通道数自动推导
    model_hw = (64, 64)
    h, w = 8, 8
    # 含scores_sum：C=4*2+1+2=11；box logit全0 -> softmax([0,0])=[0.5,0.5]
    # -> offsets=0.5 -> x1y1=(0.5-0.5)*8=0, x2y2=(0.5+0.5)*8=8 -> xyxy=[0,0,8,8]
    branch = np.zeros((1, 4 * reg_max + 1 + num_classes, h, w), dtype=np.float32)
    branch[0, 8, 0, 0] = 1.0    # scores_sum
    branch[0, 9, 0, 0] = 0.9    # cls0
    pp = build_postprocessor("dflyolo_postprocess", 0.2, 0.45, CLASS_NAMES, model_hw)
    fused = pp.multioutputs_decode([branch])
    assert len(fused) == 1 and fused[0].shape == (h * w, 5 + num_classes), fused[0].shape
    # 融合恒输出xyxy，output_layout/bbox_format被重新赋值，
    # reg_max/with_score_sum由通道数(11)与类别数(2)自动推导更新
    assert np.allclose(fused[0][0, 0:4], [0, 0, 8, 8]), fused[0][0]
    assert pp.output_layout == "grid_first" and pp.bbox_format == "xyxy"
    assert pp.reg_max == 2 and pp.with_score_sum is True
    preds = pp.decode(fused[0])
    assert np.allclose(preds[0, 0:4], [0, 0, 8, 8]), preds     # decode中bbox直通
    assert np.isclose(preds[0, 4], 0.9) and preds[0, 5] == 0, preds
    # 无scores_sum（C=4*2+2=10）：with_score_sum=False -> 手动sigmoid并补算
    # scores_sum=clip(sum(sigmoid(cls)),0,1)，融合输出为(5+nc)列
    branch_noss = np.zeros((1, 4 * reg_max + num_classes, h, w), dtype=np.float32)
    branch_noss[0, 8, 0, 0] = 0.9   # cls0（无scores_sum列，未激活logit）
    fused_noss = pp.multioutputs_decode([branch_noss])
    assert fused_noss[0].shape == (h * w, 5 + num_classes), fused_noss[0].shape
    assert pp.reg_max == 2 and pp.with_score_sum is False
    cls0_sigmoid = 1.0 / (1.0 + np.exp(-0.9))
    assert np.isclose(fused_noss[0][0, 4], 1.0), fused_noss[0][0]            # scores_sum=clip(σ0.9+σ0,0,1)=1
    assert np.isclose(fused_noss[0][0, 5], cls0_sigmoid), fused_noss[0][0]   # cls0经sigmoid
    preds_noss = pp.decode(fused_noss[0])
    assert np.isclose(preds_noss[0, 4], cls0_sigmoid) and preds_noss[0, 5] == 0, preds_noss
    # 含scores_sum（C=11）：with_score_sum=True说明ONNX本身有sigmoid节点，
    # 分数直接取用不再激活
    branch_logit = np.zeros((1, 4 * reg_max + 1 + num_classes, h, w), dtype=np.float32)
    fused_logit = pp.multioutputs_decode([branch_logit])
    assert np.allclose(fused_logit[0][0, 0:4], [0, 0, 8, 8]), fused_logit[0][0]
    assert np.isclose(fused_logit[0][0, 4], 0.0) and np.isclose(fused_logit[0][0, 5], 0.0), fused_logit[0][0]
    # 输出通道数非法（无法推导reg_max）：显式报错
    try:
        pp.multioutputs_decode([np.zeros((1, 5, h, w), dtype=np.float32)])
        raise AssertionError("非法通道数未被拦截")
    except ValueError as e:
        assert "不符" in str(e)
    # dist2bbox的cxcywh变体（网格单元尺度输出，stride还原由调用方完成）：
    # xyxy=[0.5,0.5,0.5,0.5]、cxcywh=[0.5,0.5,0,0]，乘stride后还原为像素坐标
    offsets0 = np.zeros((1, 4, 1), dtype=np.float32)      # 偏移量全0
    grid_center0 = np.array([[0.5], [0.5]], dtype=np.float32)
    stride0 = np.array([8], dtype=np.float32)
    bbox_xyxy = dist2bbox(offsets0, grid_center0) * stride0
    assert np.allclose(bbox_xyxy[0, :, 0], [4, 4, 4, 4]), bbox_xyxy
    bbox_cxcy = dist2bbox(offsets0, grid_center0, bbox_format="cxcywh") * stride0
    assert np.allclose(bbox_cxcy[0, :, 0], [4, 4, 0, 0]), bbox_cxcy
    # 3输出模型（stride 8/16/32，由model_hw与特征图hw自动推导）：
    # reshape为(bs,output_dim,H*W/64|256|1024)后按grid维concat
    b0 = np.zeros((1, 4 * reg_max + 1 + num_classes, 8, 8), dtype=np.float32)   # H*W/64=64
    b1 = np.zeros((1, 4 * reg_max + 1 + num_classes, 4, 4), dtype=np.float32)   # H*W/256=16
    b2 = np.zeros((1, 4 * reg_max + 1 + num_classes, 2, 2), dtype=np.float32)   # H*W/1024=4
    fused3 = pp.multioutputs_decode([b0, b1, b2])
    assert fused3[0].shape == (64 + 16 + 4, 5 + num_classes), fused3[0].shape
    # 行0/64/80分别为三层首个grid（xyxy格式）：x1y1=(0.5-0.5)*s=0，x2y2=s
    assert np.allclose(fused3[0][0, 0:4], [0, 0, 8, 8]), fused3[0][0]
    assert np.allclose(fused3[0][64, 0:4], [0, 0, 16, 16]), fused3[0][64]
    assert np.allclose(fused3[0][80, 0:4], [0, 0, 32, 32]), fused3[0][80]
    # 2D布局分支（导出带reshape/transpose）：stride经sqrt推导，结果一致
    fused3_2d = pp.multioutputs_decode([b0.reshape(1, 4 * reg_max + 1 + num_classes, -1),
                                        b1.reshape(1, 4 * reg_max + 1 + num_classes, -1),
                                        b2.reshape(1, 4 * reg_max + 1 + num_classes, -1)])
    assert np.allclose(fused3_2d[0], fused3[0]), fused3_2d[0]
    # 类内调用路径：postprocess遇多输出列表自动经multioutputs_decode融合，
    # 再走单输出管线（尺度还原->NMS->编码）
    b0[0, 8, 0, 0] = 1.0      # scores_sum=1
    b0[0, 9, 0, 0] = 0.9      # cls0=0.9
    outs = pp.postprocess([b0, b1, b2], [(64, 64)])
    assert len(outs[0]) == 1, outs
    assert outs[0][0]["cls_name"] == "person", outs
    assert np.isclose(outs[0][0]["score"], 0.9), outs


def test_dflyolo_multioutputs_decode_regmax1():
    """DFL多输出解码reg_max=1分支：无分布区间可加权，不做softmax加权，
    仅reshape直接输出原始回归值（通道即[l,t,r,b]四个偏移量）"""
    num_classes = len(CLASS_NAMES)
    model_hw = (64, 64)
    h, w = 8, 8
    # 含scores_sum：C=4*1+1+2=7；box通道0..3直接作为偏移量
    branch = np.zeros((1, 4 * 1 + 1 + num_classes, h, w), dtype=np.float32)
    branch[0, 0, 0, 0] = 0.5   # left
    branch[0, 2, 0, 0] = 0.5   # right
    branch[0, 4, 0, 0] = 1.0   # scores_sum
    branch[0, 5, 0, 0] = 0.9   # cls0
    pp = build_postprocessor("dflyolo_postprocess", 0.2, 0.45, CLASS_NAMES, model_hw)
    fused = pp.multioutputs_decode([branch])
    # reg_max/with_score_sum由通道数(7)与类别数(2)自动推导更新
    assert pp.reg_max == 1 and pp.with_score_sum is True
    # 偏移量直接使用：x1=(0.5-0.5)*8=0, y1=(0.5-0)*8=4,
    # x2=(0.5+0.5)*8=8, y2=(0.5+0)*8=4 -> xyxy=[0,4,8,4]
    assert np.allclose(fused[0][0, 0:4], [0, 4, 8, 4]), fused[0][0]
    assert np.isclose(fused[0][0, 4], 1.0) and np.isclose(fused[0][0, 5], 0.9), fused[0][0]
    preds = pp.decode(fused[0])
    assert np.allclose(preds[0, 0:4], [0, 4, 8, 4]), preds
    assert np.isclose(preds[0, 4], 0.9) and preds[0, 5] == 0, preds
    # 无scores_sum：C=4*1+2=6 -> reg_max=1, with_score_sum=False，
    # 手动sigmoid并补算scores_sum，bbox解码与含scores_sum形态一致
    branch_noss = np.zeros((1, 4 * 1 + num_classes, h, w), dtype=np.float32)
    branch_noss[0, 0, 0, 0] = 0.5
    branch_noss[0, 2, 0, 0] = 0.5
    branch_noss[0, 4, 0, 0] = 0.9   # cls0（无scores_sum列，未激活logit）
    fused_noss = pp.multioutputs_decode([branch_noss])
    assert pp.reg_max == 1 and pp.with_score_sum is False
    assert np.allclose(fused_noss[0][0, 0:4], [0, 4, 8, 4]), fused_noss[0][0]
    cls0_sigmoid = 1.0 / (1.0 + np.exp(-0.9))
    assert np.isclose(fused_noss[0][0, 5], cls0_sigmoid), fused_noss[0][0]


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
