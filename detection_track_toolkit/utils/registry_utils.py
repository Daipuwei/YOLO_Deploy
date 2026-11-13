# -*- coding: utf-8 -*-
# @Time    : 2026/9/9 01:04
# @Author  : DaiPuWei
# @Email   : daipuwei@qq.com
# @File    : registry_utils.py
# @Software: PyCharm

"""
    这是定义注册器类的脚本
    基于注册器模式提供名称到对象的映射，实现工厂与实现的解耦：
        1.开闭原则：新增模型/引擎等模块时，只需在模块内注册，工厂无需改动；
        2.双模式注册：register()既可作为装饰器使用，也可直接函数调用，并支持
          显式指定注册名（别名）；
        3.注册约束：重名注册显式抛错（python -O下同样生效），防止意外覆盖；
        4.查询与维护：get()按名称获取（未注册时提示检查import），并支持
          keys()/unregister()/in/len等查询与移除操作。

    参考：https://github.com/JDAI-CV/fast-reid/blob/master/fastreid/utils/registry.py
"""

from typing import Dict, Optional


class Registry:
    """
    这是定义注册器的类，提供名称到对象的映射，支持第三方扩展自定义模块
    用法：
        MODEL_REGISTRY = Registry('MODEL')              # 创建注册器
        @MODEL_REGISTRY.register()                      # 装饰器方式注册
        class YOLOv8():
            ...
        MODEL_REGISTRY.register(MyModel)                # 函数调用方式注册
        MODEL_REGISTRY.register(MyModel, name='v8')     # 显式指定注册名（别名）
    """

    def __init__(self, name: str, doc: str = None) -> None:
        """
        这是初始化注册器的函数
        Args:
            name: 注册器名称（报错信息中标识归属）
            doc: 注册器用途说明（可选，写入实例__doc__，供help()展示）
        Returns:
        """
        self._name: str = name
        self._obj_map: Dict[str, object] = {}
        if doc:
            self.__doc__ = doc

    def _do_register(self, name: str, obj: object) -> None:
        # 重名注册直接报错，防止意外覆盖已注册对象（不用assert：python -O下断言会被剥离）
        if name in self._obj_map:
            raise KeyError("An object named '{}' was already registered in '{}' registry!"
                           .format(name, self._name))
        self._obj_map[name] = obj

    def register(self, obj: object = None, name: str = None) -> Optional[object]:
        """
        这是注册对象的函数，支持装饰器与函数调用两种方式
        Args:
            obj: 被注册的对象（装饰器方式时为None）
            name: 注册名，缺省取对象的__name__，可显式指定别名
        Returns:
            被注册的对象（装饰器方式返回装饰器）
        """
        if obj is None:
            # 装饰器方式注册
            def deco(func_or_class: object) -> object:
                self._do_register(name or func_or_class.__name__, func_or_class)
                return func_or_class

            return deco

        # 函数调用方式注册，返回被注册对象以支持链式调用
        self._do_register(name or obj.__name__, obj)
        return obj

    def get(self, name: str) -> object:
        """
        这是按名称获取已注册对象的函数
        Args:
            name: 注册名
        Returns:
            已注册对象
        """
        if name not in self._obj_map:
            raise KeyError(
                "No object named '{}' found in '{}' registry!"
                "（如该对象已定义，请确认其所在模块已被 import）".format(name, self._name)
            )
        return self._obj_map[name]

    def keys(self) -> list:
        """
        这是获取全部已注册名称的函数
        Returns:
            已注册名称列表
        """
        return list(self._obj_map.keys())

    def unregister(self, name: str) -> object:
        """
        这是移除已注册对象的函数（名称不存在时抛KeyError）
        Args:
            name: 注册名
        Returns:
            被移除的对象
        """
        return self._obj_map.pop(name)

    def __contains__(self, name: str) -> bool:
        """
        这是判断名称是否已注册的函数，支持in运算
        Args:
            name: 注册名
        Returns:
            是否已注册
        """
        return name in self._obj_map

    def __len__(self) -> int:
        """
        这是获取已注册对象个数的函数，支持len运算
        Returns:
            已注册对象个数
        """
        return len(self._obj_map)
