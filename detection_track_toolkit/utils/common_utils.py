# -*- coding: utf-8 -*-
# @Time    : 2026/9/10 08:59
# @Author  : DaiPuWei
# @Email   : daipuwei@qq.com
# @File    : common_utils.py
# @Software: PyCharm

"""
    这是定义公共工具的脚本。

    模块内容：
        1. ArgsParser：命令行参数解析基类，预置-o/--opt配置覆盖参数；
           _parse_opt把"key=value"经yaml解析为带类型的值（键按第一个'='
           分割，配置值本身可含'='；重复键告警后覆盖）；
        2. load_yaml(yaml_path)：加载yaml配置文件的函数
           （文件不存在或解析失败时带路径明确报错）；
        3. save_yaml(cfg, yaml_path)：配置参数字典写入yaml文件的函数
           （自动创建目录，utf-8编码）；
        4. merge_config(cfg, opt)：命令行覆盖参数合并进配置字典的函数
           （就地修改cfg并返回）：顶层键直接覆盖（值为字典且键已存在时
           合并更新），点分键a.b.c深度写入，路径不存在时明确报错；
        5. init_config(args)：加载yaml并合并命令行覆盖的配置入口；
        6. print_error(value)：多进程任务错误回调函数
           （供video2voc_dataset等脚本的Pool.apply_async使用）。
"""

import os
import yaml

# 仓库根目录（包目录上一级）：配置yaml内相对路径（./model_data/...、./logs/...）
# 的兜底解析锚点，配置模板中的相对路径语义均以仓库根为基准书写
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", ".."))


def resolve_config_path(path):
    """
    这是解析配置内文件路径的函数（配置相对路径的统一解析口径）：
    绝对路径原样返回；相对路径优先按进程CWD解析（兼容自定义工程的
    相对路径用法），CWD下不存在时回退锚定仓库根（REPO_ROOT，包目录上一级）。
    修复背景：配置模板（如./model_data/xxx.onnx、./logs/xxx.log）曾一律按
    进程CWD解析，从非仓库根目录运行工具脚本时模型/类别名/日志路径
    全部FileNotFoundError，违背tools脚本"任意cwd可运行"的设计目标；
    改为CWD优先+仓库根兜底后既保持向后兼容，又使任意cwd运行成立。
    Args:
        path: 配置内的文件路径（绝对或相对）
    Returns:
        解析后的绝对路径（不做存在性校验，由各消费方自行校验并给出
        面向自身语义的报错信息）
    """
    # 参数校验：None/空串无明确路径语义（空串会经os.path.exists("")为False
    # 静默解析为仓库根目录、None则os.path.isabs抛TypeError），显式报错
    # 给出清晰定位（修复：曾静默返回仓库根/抛原生TypeError）
    if not path:
        raise ValueError("配置路径不能为None或空字符串")
    if os.path.isabs(path):
        return path
    # CWD优先：自定义工程依赖CWD相对路径的既有用法保持不变
    if os.path.exists(path):
        return os.path.abspath(path)
    # 仓库根兜底：配置模板相对路径在任意cwd下均可解析
    # （abspath归一化拼接后路径中的"./"等冗余段，保证返回规范绝对路径）
    return os.path.abspath(os.path.join(REPO_ROOT, path))

from argparse import ArgumentParser, RawDescriptionHelpFormatter

class ArgsParser(ArgumentParser):

    def __init__(self):
        """
        这是自定义命令行参数解析类的初始化函数
        """
        super(ArgsParser, self).__init__(
            formatter_class=RawDescriptionHelpFormatter)
        self.add_argument(
            "-o", "--opt", nargs='+', help="set configuration options")

    def parse_args(self, argv=None):
        """
        这是解析命令行参数的函数
        Args:
            argv: 命令行参数列表，默认为None
        Returns:
            解析后的参数对象（opt键为"key=value"解析出的配置字典）
        """
        args = super(ArgsParser, self).parse_args(argv)
        args.opt = self._parse_opt(args.opt)
        return args

    def _parse_opt(self, opts):
        """
        这是把"-o key=value"参数解析为配置字典的函数
        Args:
            opts: -o参数值列表
        Returns:
            配置字典（值经yaml解析为带类型的数据；
            on/off/yes/no/y/n等YAML 1.1布尔易混值按字符串处理并告警）
        """
        config = {}
        if not opts:
            return config
        for s in opts:
            s = s.strip()
            # 按第一个'='分割：配置值本身可能含'='
            k, v = s.split('=', 1)
            if k in config:
                print("警告：配置键'{0}'重复指定，后者覆盖前者".format(k))
            value = yaml.load(v, Loader=yaml.FullLoader)
            # YAML 1.1布尔陷阱（修复：on/off/yes/no/y/n及大小写变体曾被
            # 静默解析为bool，如model_type=on变成True；本工具集-o值域为
            # 数值/路径/枚举字符串/True/False，此类易混值保持字符串并告警，
            # 布尔请显式写true/false）
            if isinstance(value, bool) and \
                    v.strip().strip('\'"').lower() not in ("true", "false"):
                print("[WARN] -o参数{0}的值'{1}'与YAML 1.1布尔字面量同形，"
                      "已按字符串处理（布尔值请显式写true/false）".format(k, v))
                value = v
            config[k] = value
        return config

def load_yaml(yaml_path):
    """
    这是加载yaml文件的函数
    Args:
        yaml_path: yaml文件路径
    Returns:
        配置参数字典
    """
    yaml_path = os.path.abspath(yaml_path)
    if not os.path.exists(yaml_path):
        raise FileNotFoundError("配置文件不存在：{0}".format(yaml_path))
    try:
        with open(yaml_path, 'r') as f:
            cfg = yaml.load(f, Loader=yaml.FullLoader)
    except yaml.YAMLError as msg:
        raise ValueError("配置文件解析失败：{0}：{1}".format(yaml_path, msg))
    return cfg

def save_yaml(cfg,yaml_path):
    """
    这是配置参数字典写入yaml文件的函数
    Args:
        cfg: 配置参数字典
        yaml_path: yaml文件路径
    Returns:
    """
    yaml_dir = os.path.dirname(yaml_path)
    if yaml_dir and not os.path.exists(yaml_dir):
        os.makedirs(yaml_dir)
    with open(yaml_path, 'w', encoding='utf-8') as f:
        yaml.dump(cfg,f)

def merge_config(cfg,opt):
    """
    这是将命令行解析参数合并到参数字典的函数（就地修改cfg并返回）。
    路径健壮性约定（修复：曾对中间层非dict以原生TypeError崩溃、
    两层路径末级键拼错时静默新建键使覆盖完全不生效）：
    - 点号路径逐级下钻：每一级中间层必须存在且为dict（否则ValueError
      指明路径与实际类型）；
    - 末级键必须已存在（拼错立即报错并列出同级候选键，不静默新建）；
    - 无点号顶层键：值为dict且目标存在时update（目标非dict报错），
      其余情况直接赋值（允许新建顶层键，兼容追加整段配置）。
    Args:
        cfg: 参数字典
        opt: 命令行参数解析字典
    Returns:
        合并后的参数字典（与cfg为同一对象）
    """
    for key,value in opt.items():
        if "." not in key:
            if isinstance(value, dict) and key in cfg.keys():
                if not isinstance(cfg[key], dict):
                    raise ValueError("顶层键'{0}'的现有值为{1}（非字典），"
                                     "无法与字典值合并，请检查配置"
                                     .format(key, type(cfg[key]).__name__))
                cfg[key].update(value)
            else:
                cfg[key] = value
            continue
        sub_keys = key.split('.')
        if sub_keys[0] not in cfg.keys():
            raise ValueError("配置键'{0}'的顶层键'{1}'不存在，可用顶层键为{2}"
                             .format(key, sub_keys[0], list(cfg.keys())))
        cur = cfg[sub_keys[0]]
        # 下钻中间层（除末级键外）：必须存在且为dict
        for idx, sub_key in enumerate(sub_keys[1:-1]):
            if not isinstance(cur, dict) or sub_key not in cur:
                raise ValueError("配置键'{0}'的路径'{1}'不存在或中间层非字典"
                                 "（实际类型{2}），可用键为{3}"
                                 .format(key, '.'.join(sub_keys[:idx + 2]),
                                         type(cur).__name__,
                                         list(cur.keys()) if isinstance(cur, dict) else cur))
            cur = cur[sub_key]
        # 末级键：必须已存在（拼错立即报错，不静默新建）
        last_key = sub_keys[-1]
        if not isinstance(cur, dict):
            raise ValueError("配置键'{0}'的父层'{1}'为{2}（非字典），无法设置子键"
                             .format(key, '.'.join(sub_keys[:-1]), type(cur).__name__))
        if last_key not in cur:
            raise ValueError("配置键'{0}'的末级键'{1}'不存在（不静默新建，"
                             "请检查拼写），可用键为{2}"
                             .format(key, last_key, list(cur.keys())))
        cur[last_key] = value
    return cfg

def init_config(args):
    """
    这是根据命令行初始化参数的函数
    Args:
        args: 命令行解析类
    Returns:
        合并命令行覆盖后的配置参数字典
    """
    # 加载yaml文件参数
    cfg = load_yaml(args.cfg)
    # 合并更新参数
    cfg = merge_config(cfg,args.opt)
    return cfg

def print_error(value):
    """
    这是多进程任务错误回调的函数（供Pool.apply_async使用）
    Args:
        value: 任务异常信息
    Returns:
    """
    print("error: ", value)
