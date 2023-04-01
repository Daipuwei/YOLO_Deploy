# -*- coding: utf-8 -*-
# @Time    : 2023/3/28 下午2:34
# @Author  : DaiPuWei
# @Email   : daipuwei@qq.com
# @File    : json_utils.py
# @Software: PyCharm

"""
    这是JSON相关工具和工具类的定义脚本
"""

import json
import numpy as np

class NpEncoder(json.JSONEncoder):

    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return list(obj)
        elif isinstance(obj,bytes):
            return str(obj,encoding='utf-8')
        else:
            return super(NpEncoder, self).default(obj)