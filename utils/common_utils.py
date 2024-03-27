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
from argparse import ArgumentParser, RawDescriptionHelpFormatter

class ArgsParser(ArgumentParser):

    def __init__(self):
        """
            这是自定义命令行参数解析类的函数
        """
        super(ArgsParser, self).__init__(
            formatter_class=RawDescriptionHelpFormatter)
        self.add_argument(
            "-o", "--opt", nargs='+', help="set configuration options")
        self.add_argument(
            '-p',
            '--profiler_options',
            type=str,
            default=None,
            help='The option of profiler, which should be in format ' \
                 '\"key1=value1;key2=value2;key3=value3\".'
        )

    def parse_args(self, argv=None):
        args = super(ArgsParser, self).parse_args(argv)
        args.opt = self._parse_opt(args.opt)
        return args
    def _parse_opt(self, opts):
        config = {}
        if not opts:
            return config
        for s in opts:
            s = s.strip()
            k, v = s.split('=')
            config[k] = yaml.load(v, Loader=yaml.Loader)
        return config

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

def save_yaml(cfg,yaml_path):
    """
    这是配置参数字典写入yaml文件的函数
    Args:
        cfg: 配置参数字典
        yaml_path: yaml文件路径
    Returns:
    """
    with open(yaml_path, 'w') as f:
        yaml.dump(cfg,f)

def merge_config(cfg,opt):
    """
    这是将命令行解析参数合并到参数字典的函数
    Args:
        cfg: 参数字典
        opt: 命令行参数解析类
    Returns:
    """
    #print(opt)
    for key,value in opt.items():
        if "." not in key:
            if isinstance(value, dict) and key in cfg.keys():
                cfg[key].update(value)
            else:
                cfg[key] = value
        else:
            sub_keys = key.split('.')
            assert (
                sub_keys[0] in cfg.keys()
            ), "the sub_keys can only be one of global_config: {}, but get: " \
               "{}, please check your running command".format(
                cfg.keys(), sub_keys[0])
            cur = cfg[sub_keys[0]]
            for idx, sub_key in enumerate(sub_keys[1:]):
                if idx == len(sub_keys) - 2:
                    cur[sub_key] = value
                else:
                    cur = cur[sub_key]
    return cfg

def init_config(args):
    """
    这是根据命令行初始化参数的函数
    Args:
        args: 命令行解析类
    Returns:
    """
    # 加载yaml文件参数
    cfg = load_yaml(args.cfg)
    # 合并更新参数
    cfg = merge_config(cfg,args.opt)
    return cfg

def print_error(value):
    """
    定义错误回调函数
    Args:
        value:
    Returns:
    """
    print("error: ", value)
