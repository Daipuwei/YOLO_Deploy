# -*- coding: utf-8 -*-
"""
    utils通用工具单元测试：merge_config路径健壮性、_parse_opt的YAML布尔
    陷阱、utils/__init__惰性导出的依赖缺失区分
"""

import os
import sys
import importlib

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from detection_track_toolkit.utils.common_utils import merge_config
from detection_track_toolkit.utils.common_utils import ArgsParser


def test_merge_config_valid_paths():
    """合法覆盖路径：顶层赋值/顶层dict合并/两级/三级路径正常生效"""
    cfg = {"DetectionModel": {"model_type": "yolov5", "threshold": 0.5,
                              "nested": {"a": 1}},
           "Log": {"log_path": "a.log"}}
    merge_config(cfg, {"DetectionModel.threshold": 0.6,
                       "DetectionModel.nested.a": 2,
                       "Log": {"log_max_bytes": 100},
                       "NewTop": 1})
    assert cfg["DetectionModel"]["threshold"] == 0.6
    assert cfg["DetectionModel"]["nested"]["a"] == 2
    assert cfg["Log"]["log_max_bytes"] == 100
    assert cfg["NewTop"] == 1


def test_merge_config_intermediate_not_dict():
    """中间层非dict显式报错（回归：曾原生TypeError 'int' object does not
    support item assignment，报错信息与配置语义脱节）"""
    cfg = {"Log": 5}
    try:
        merge_config(cfg, {"Log.log_path": "x"})
        assert False, "中间层非dict应报错"
    except ValueError as e:
        assert "Log" in str(e), e
    cfg = {"A": {"b": 5}}
    try:
        merge_config(cfg, {"A.b.c": 1})
        assert False, "中间层非dict应报错"
    except ValueError as e:
        assert "A.b" in str(e), e


def test_merge_config_last_key_typo_raises():
    """末级键拼错立即报错（回归：两层路径曾静默新建键，覆盖完全不生效）"""
    cfg = {"DetectionModel": {"model_type": "yolov5"}}
    try:
        merge_config(cfg, {"DetectionModel.model_typ": 8})
        assert False, "末级键拼错应报错"
    except ValueError as e:
        assert "model_typ" in str(e) and "model_type" in str(e), e
    # 三层路径末级拼错同样报错
    cfg = {"A": {"b": {"c": 1}}}
    try:
        merge_config(cfg, {"A.b.c2": 1})
        assert False, "三层末级拼错应报错"
    except ValueError as e:
        assert "c2" in str(e), e


def test_parse_opt_yaml_bool_trap():
    """YAML 1.1布尔易混值按字符串处理（回归：model_type=on曾被解析为True）；
    true/false仍正常解析为bool"""
    parser = ArgsParser()
    config = parser._parse_opt(["model_type=on", "name=n", "flag=Yes",
                                "is_normalize=true", "is_normalize2=False",
                                "path=./a.log", "num=3"])
    assert config["model_type"] == "on", config
    assert config["name"] == "n", config
    assert config["flag"] == "Yes", config
    assert config["is_normalize"] is True
    assert config["is_normalize2"] is False
    assert config["path"] == "./a.log"
    assert config["num"] == 3


def test_utils_lazy_import_dependency_missing():
    """utils惰性导出区分异常来源（回归：子模块依赖缺失曾被吞掉，
    报出误导性AttributeError）：
    - 缺失的是子模块本身（已迁移）→ 跳过，最终AttributeError
    - 缺失的是子模块的第三方依赖 → 原样抛ModuleNotFoundError"""
    import detection_track_toolkit.utils as utils_pkg
    real_import = importlib.import_module
    try:
        # 模拟：探测registry_utils时抛"pycocotools缺失"（依赖缺失）→ 原样抛出
        def fake_import(name, package=None):
            if name == ".registry_utils":
                raise ModuleNotFoundError("No module named 'pycocotools'",
                                          name="pycocotools")
            return real_import(name, package)
        importlib.import_module = fake_import
        try:
            getattr(utils_pkg, "Registry")
            assert False, "依赖缺失应原样抛ModuleNotFoundError"
        except ModuleNotFoundError as e:
            assert e.name == "pycocotools", e.name
        finally:
            # 场景间恢复真实import（否则场景2探测registry_utils时再次被拦截）
            importlib.import_module = real_import
        # 场景2：子模块本身不存在（已迁移/删除）→ 跳过，最终AttributeError。
        # 注入真实不存在的子模块名（走真实import路径自然抛
        # ModuleNotFoundError(name=该子模块)，不用mock——mock掉真实子模块
        # 会使其他子模块的from-import嵌套触发包装成ImportError）
        saved_subs = utils_pkg._SUBMODULES
        try:
            utils_pkg._SUBMODULES = saved_subs + ("definitely_gone",)
            try:
                getattr(utils_pkg, "DefinitelyNotHere")
                assert False, "不存在的符号应AttributeError"
            except AttributeError:
                pass
        finally:
            utils_pkg._SUBMODULES = saved_subs
    finally:
        importlib.import_module = real_import


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
