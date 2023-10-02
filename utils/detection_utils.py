# -*- coding: utf-8 -*-
# @Time    : 2023/3/28 下午2:35
# @Author  : DaiPuWei
# @Email   : daipuwei@qq.com
# @File    : detection_utils.py
# @Software: PyCharm

"""
    这是定义检测模型相关工具的脚本
"""

import cv2
import numpy as np

def get_classes(classes_path):
    """
    这是获取目标分类名称的函数
    :param classes_path: 目标分类名称txt文件路径
    :return:
    """
    classes_names = []
    with open(classes_path, 'r') as f:
        for line in f.readlines():
            classes_names.append(line.strip())
    # classes_names = np.array(classes_names)
    return classes_names

def letterbox(image, resize_shape=(640, 640), color=(114, 114, 114)):
    h, w, c = image.shape
    input_h,input_w = resize_shape
    # Calculate widht and height and paddings
    r = min(input_w / w,input_h / h)
    new_unpad = int(round(w * r)), int(round(h * r))
    dw,dh = input_w-new_unpad[0],input_h-new_unpad[1]
    dw /= 2
    dh /= 2
    resize_image = cv2.resize(image, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    resize_image = cv2.copyMakeBorder(resize_image, top, bottom, left, right,
                                      cv2.BORDER_CONSTANT, value=color)  # add border
    #cv2.imwrite('letterbox.jpg', resize_image)
    return resize_image

def xywh2xyxy(x):
    """
    这是转换预测框坐标格式的函数，xywh->xyxy
    Args:
        x: 预测框数组，shape为(n,4),
    Returns:
    """
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y

def scale_coords(coords,img1_shape, img0_shape):
    gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
    i2d = np.array([[gain,0,(-gain*img0_shape[1]+img1_shape[1]+gain-1)*0.5],
                    [0,gain,(-gain*img0_shape[0]+img1_shape[0]+gain-1)*0.5]])
    d2i = cv2.invertAffineTransform(i2d)
    d2i = np.reshape(d2i,(6,1))
    #pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shap,1e[0] - img0_shape[0] * gain) / 2  # wh padding
    # gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
    # pad = (img0_shape[1] - img1_shape[1] / gain) / 2, (img0_shape[0] - img1_shape[0] / gain) / 2  # wh padding


    # coords[:, :4] /= gain
    # coords[:, [0, 2]] -= pad[0]  # x padding
    # coords[:, [1, 3]] -= pad[1]  # y padding
    coords[:, 0] = d2i[0]*coords[:,0] + d2i[2]
    coords[:, 1] = d2i[0]*coords[:, 1]+d2i[5]
    coords[:, 2] = d2i[0]*coords[:,2] + d2i[2]
    coords[:, 3] = d2i[0]*coords[:, 3]+d2i[5]

    clip_coords(coords, img0_shape)
    return coords


def clip_coords(bboxes, img_shape):
    """
    这是对检测结果进行截断的函数，防止出现坐标越界情况
    Args:
        boxes: 检测框
        img_shape: 图像尺度大小
    Returns:
    """
    bboxes[:, 0] = np.clip(bboxes[:, 0], 0, img_shape[1])   # x1
    bboxes[:, 1] = np.clip(bboxes[:, 1], 0, img_shape[0])   # y1
    bboxes[:, 2] = np.clip(bboxes[:, 2], 0, img_shape[1])   # x2
    bboxes[:, 3] = np.clip(bboxes[:, 3], 0, img_shape[0])   # y2
    return bboxes

def draw_detection_results(image,preds,classes_names,colors):
    """
    这是绘制人脸检测结果的函数
    Args:
        image: 图像，opnecv读入
        preds: 检测结果，shape为(num,5)
        classes_names: 目标名称列表
        colors: rgb颜色列表
    Returns:
    """
    h,w,_= np.shape(image)
    tl = min(round((image.shape[0] + image.shape[1]) // 300),1)  # line/font thickness
    # print(np.shape(preds))
    #print(preds)
    for x1, y1, x2, y2, score,cls_id in preds:
        x1 = int(round(x1))
        y1 = int(round(y1))
        x2 = int(round(x2))
        y2 = int(round(y2))
        cls_id = int(cls_id)
        #print(x1,y1,x2,y2)
        text = "{0}:{1:.4f}".format(classes_names[cls_id],score)
        tf = max(tl - 1, 1)
        t_size = cv2.getTextSize(text, 0, fontScale=tl / 3, thickness=tf)[0]
        cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), colors[cls_id], thickness=tl, lineType=cv2.LINE_AA)
        cv2.rectangle(image, (int(x1), int(y1)), (int(x1 + t_size[0]), int(y1 - t_size[1] - 3)), colors[cls_id],
                      thickness=-1, lineType=cv2.LINE_AA)
        cv2.putText(image, text, (int(x1), int(y1 - 2)), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=tl / 3,
                    color=(255, 255, 255), thickness=tf, lineType=cv2.LINE_AA)
    return image
