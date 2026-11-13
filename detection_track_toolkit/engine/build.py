# -*- coding: utf-8 -*-
# @Time    : 2026/9/9 06:53
# @Author  : DaiPuWei
# @Email   : daipuwei@qq.com
# @File    : build.py
# @Software: PyCharm

"""
    这是定义推理引擎类注册创建函数的脚本。

    模块内容：
        1. ENGINE_REGISTRY：推理引擎注册器，各引擎工厂函数（如onnxruntime、tensorrt）
           通过装饰器注册，注册名即工厂函数名；
        2. _LAZY_ENGINE_MODULES：引擎目录下可惰性导入的子包白名单（动态发现），
           非引擎工具包经 _NON_ENGINE_PACKAGES 排除；
        3. _ENGINE_MODULE_MAP：注册名→引擎包名的显式映射，包名与注册名不一致时登记；
        4. build_engine(logger, cfg, registry=None, **kwargs)：引擎构建入口：
           - cfg 必须含 engine_type 键（缺失/未知分别报ValueError并列出可用引擎），
             取值统一转小写匹配注册名；
           - 未注册时按白名单惰性导入对应引擎包触发注册（重库延迟到首次使用），
             引擎包 __init__ 约定只做注册、不执行重操作；
           - registry 缺省使用模块级 ENGINE_REGISTRY，测试/隔离场景可注入；
           - kwargs 透传给引擎工厂（gpu_id由工厂读取，其余键注入引擎属性，
             冲突时 BaseEngine 抛ValueError）。
"""

import importlib
import os
import pkgutil
from typing import Any, Dict, Optional

from ..utils import Registry

ENGINE_REGISTRY: Registry = Registry(
    "ENGINE",
    doc="这是推理引擎类的注册器，用于自动生成不同推理引擎类，例如onnxruntime、tensorrt")

# 引擎目录下的可用子包（白名单）：build_engine 惰性注册时只允许导入这些模块，
# 既避免 import engine 时拉起 onnxruntime/tensorrt 等重库，也防止配置注入任意模块路径。
# 若目录中出现非引擎工具包，加入 _NON_ENGINE_PACKAGES 排除，
# 避免它出现在"可用引擎"列表或被 engine_type 误触发导入。
_ENGINE_PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))
_NON_ENGINE_PACKAGES = frozenset()   # 例如 {"common"} 之类的工具包。
_LAZY_ENGINE_MODULES = {m.name for m in pkgutil.iter_modules([_ENGINE_PACKAGE_DIR])
                        if m.ispkg and m.name not in _NON_ENGINE_PACKAGES}

# 注册名→引擎包名的显式映射：约定引擎包名与注册名一致（无需登记）；
# 两者不一致时在此登记，惰性导入按映射定位模块，而不是静默失效。
_ENGINE_MODULE_MAP: Dict[str, str] = {}


def build_engine(logger: Any, cfg: Dict[str, Any],
                 registry: Optional[Registry] = None, **kwargs: Any) -> Any:
    """
    这是初始化推理引擎的函数。

    Args:
        logger: 日志类实例
        cfg: 参数配置字典（engine_type必填；engine_model_path等其余键由对应引擎工厂消费）
        registry: 引擎注册器，缺省使用模块级ENGINE_REGISTRY
                  （测试/隔离场景可注入自定义注册表）
        kwargs: 自定义参数字典（透传给对应引擎工厂，支持的键：
                gpu_id——GPU设备号，各引擎工厂读取、缺省0；
                其余键透传给引擎类并注入为属性，与已有属性冲突时BaseEngine抛ValueError）
    Returns:
        推理引擎类实例
    """
    registry = ENGINE_REGISTRY if registry is None else registry
    # engine_type缺失与未知引擎分开报错：'None'字样会让排查者误以为配置写成了None。
    engine_type = cfg.get("engine_type")
    if engine_type is None:
        raise ValueError("配置中缺少engine_type键，可用引擎：{0}"
                         .format(_available_engines(registry)))
    # 引擎类型统一转小写匹配注册名（注册名即工厂函数名，约定全小写）。
    engine_type = str(engine_type).lower()
    # 未注册时按白名单惰性导入对应引擎包触发注册（引擎包 __init__ 约定只做注册、
    # 不执行重操作；重库延迟到首次真正使用该引擎时加载），仍无则报错列出可用引擎。
    if engine_type not in registry:
        module_name = _ENGINE_MODULE_MAP.get(engine_type, engine_type)
        if module_name in _LAZY_ENGINE_MODULES:
            importlib.import_module("." + module_name, __package__)
    if engine_type not in registry:
        # 配置错误语义用ValueError（KeyError偏"查找失败"）。
        raise ValueError("未知的引擎类型 '{0}'，可用引擎：{1}"
                         .format(engine_type, _available_engines(registry)))
    return registry.get(engine_type)(logger, cfg, **kwargs)


def _available_engines(registry: Registry):
    """计算可用引擎列表。

    可用引擎 = 已注册的 + 白名单内可惰性导入的 + 映射表登记的，
    避免注册表为空时列出空列表误导排查。
    """
    return sorted(set(registry.keys()) | _LAZY_ENGINE_MODULES | set(_ENGINE_MODULE_MAP))
