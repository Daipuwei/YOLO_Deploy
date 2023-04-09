# -*- coding: utf-8 -*-
# @Time    : 2023/3/28 下午2:35
# @Author  : DaiPuWei
# @Email   : daipuwei@qq.com
# @File    : common_utils.py
# @Software: PyCharm

"""
    这是定义公共工具的脚本
"""


import os
import yaml

def load_yaml(yaml_path):
    """
    这是加载ymal文件的函数
    Args:
        yaml_path: yaml文件路径
    Returns:
    """
    with open(os.path.abspath(yaml_path), 'r') as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    return cfg

def print_error(value):
    """
    定义错误回调函数
    Args:
        value:
    Returns:
    """
    print("error: ", value)
