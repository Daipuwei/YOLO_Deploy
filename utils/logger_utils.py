# -*- coding: utf-8 -*-
# @Time    : 2023/3/28 下午2:34
# @Author  : DaiPuWei
# @Email   : daipuwei@qq.com
# @File    : logger_utils.py
# @Software: PyCharm

"""
    这是定义日志相关工具的脚本
"""

import logging

def logger_config(log_path,logging_name):
    """
    这是配置日志实例的函数
    Args:
        log_path: 日志文件路径
        logging_name: 记录中name，可随意
    Returns:
    """
    # 获取logger对象,取名
    logger = logging.getLogger(logging_name)
    # 输出DEBUG及以上级别的信息，针对所有输出的第一层过滤
    logger.setLevel(level=logging.DEBUG)
    # 获取文件日志句柄并设置日志级别，第二层过滤
    handler = logging.FileHandler(log_path, encoding='UTF-8')
    handler.setLevel(logging.INFO)
    # 生成并设置文件日志格式
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    # console相当于控制台输出，handler文件输出。获取流句柄并设置日志级别，第二层过滤
    console = logging.StreamHandler()
    console.setLevel(logging.DEBUG)
    # 为logger对象添加句柄
    logger.addHandler(handler)
    logger.addHandler(console)
    return logger