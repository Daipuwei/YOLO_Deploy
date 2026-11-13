# -*- coding: utf-8 -*-
# @Time    : 2026/9/8 11:48
# @Author  : DaiPuWei
# @Email   : daipuwei@qq.com
# @File    : logger_utils.py
# @Software: PyCharm

"""
    这是定义全局日志管理器的脚本
    基于单例类Logger提供统一的日志入口，配置、写入、等级与生命周期收拢于一处：
        1.单实例保障：进程内所有模块共享同一个Logger对象；
        2.生命周期显式化：configure()（幂等装配）/ reconfigure()（强制重建）/
          complete()（队列排空）/ shutdown()（排空并摘除）/ reset()（状态还原）；
        3.等级调节：set_level()动态调整全局等级，文件与控制台sink可分别设置级别，
          两者同时满足才会落盘；
        4.等级日志函数：debug/info/warning/error/critical/exception 由Logger
          显式提供并委托loguru实现，不向外暴露loguru实例；
        5.入口导出：init_logger()与logger一并提供，logger即Logger单例，可直接记录日志。

    单例实现要点：
        1.__new__双检锁：多线程同时首次实例化也只产生一个实例；
        2.陷阱：每次Logger()都会执行__init__，若在__init__里初始化状态会被反复重置，
          因此实例状态只在__new__里初始化一次；
        3.单例边界是进程：fork后子进程拥有独立实例（与loguru全局logger同理）。
"""

import os
import sys
import glob
import logging
import threading

from multiprocessing import Process
from loguru import logger as _loguru_logger

from .common_utils import resolve_config_path

__all__ = ['Logger', 'init_logger', 'get_logger', 'get_configured_path',
           'set_global_level', 'logger']

# 默认日志路径（cfg缺键且未显式传参时使用）
DEFAULT_LOG_PATH = './logs/yolo_deploy.log'
# 默认单文件大小上限：50MB，达到后轮转
DEFAULT_MAX_BYTES = 50 * 1024 * 1024
# 默认备份文件保留个数
DEFAULT_BACKUP_COUNT = 7
# 默认文件与控制台sink级别
DEFAULT_FILE_LEVEL = 'DEBUG'
DEFAULT_CONSOLE_LEVEL = 'INFO'
# 默认全局等级（全局等级过滤，运行时可用set_level动态调整）
DEFAULT_GLOBAL_LEVEL = 'DEBUG'
# 文件日志格式：时间、级别、模块、函数、行号与消息
FILE_FORMAT = ('{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | '
               '{name}:{function}:{line} - {message}')
# 控制台日志格式：彩色输出，级别字段随级别自动变色
CONSOLE_FORMAT = ('<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | '
                  '<level>{level: <8}</level> | '
                  '<cyan>{name}</cyan>:<cyan>{function}</cyan> - {message}')


def _bytes_to_rotation(max_bytes):
    """
    这是将字节数转换为loguru轮转字符串的函数
    Args:
        max_bytes: 单文件大小上限（字节）
    Returns:
        轮转字符串（≥1MB取整到MB，否则取整到KB，最小1单位）
    """
    if max_bytes >= 1024 * 1024:
        return '{0} MB'.format(max(1, int(max_bytes) // (1024 * 1024)))
    return '{0} KB'.format(max(1, int(max_bytes) // 1024))


class InterceptHandler(logging.Handler):
    """
    这是定义stdlib logging桥接器的类
    把tqdm、onnxruntime等基于标准库logging的第三方日志记录转发进loguru，
    使其与业务日志进入同一文件
    """

    def emit(self, record):
        """
        这是转发stdlib日志记录到loguru的函数
        Args:
            record: stdlib日志记录
        Returns:
        """
        # 将stdlib级别名转换为loguru级别，自定义级别则直接使用级别号
        try:
            level = _loguru_logger.level(record.levelname).name
        except ValueError:
            level = record.levelno
        # 回溯跳过logging自身的栈帧，让日志中的文件与行号指向真正的调用点
        frame, depth = logging.currentframe(), 2
        while frame.f_code.co_filename == logging.__file__:
            frame = frame.f_back
            depth += 1
        _loguru_logger.opt(depth=depth, exception=record.exc_info).log(
            level, record.getMessage())


class Logger:
    """
    这是定义日志管理器单例的类
    负责日志的配置、重建、排空、关停与状态查询
    用法：
        Logger().configure(cfg)              # 幂等初始化
        Logger().info('...')                 # 直接记录日志（无需暴露实例）
        Logger().error('...')                # 各等级均有对应函数
        Logger().reconfigure(log_path='...') # 强制换路径/级别
        Logger().complete()                  # 异步队列排空（优雅退出时调用）
        Logger().shutdown()                  # 排空并摘除全部sink
        Logger().set_level('ERROR')          # 动态调整全局等级（无需重建）
        Logger.reset()                       # 测试用：恢复未初始化状态
    """

    _instance = None
    _instance_lock = threading.Lock()

    def __new__(cls):
        # 双检锁单例：多线程同时首次实例化也只产生一个实例
        # 实例状态只在__new__里初始化一次，放在__init__会在每次构造时被重置（经典单例陷阱）
        if cls._instance is None:
            with cls._instance_lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._init_lock = threading.Lock()
                    cls._instance._configured_path = None
                    cls._instance._file_sink_id = None
                    cls._instance._console_sink_id = None
                    cls._instance._intercept_installed = False
                    # 全局等级状态：可变容器，sink的filter函数动态读取，改等级无需重建sink
                    # 初始值也解析为Level对象，与set_level的存储格式一致
                    cls._instance._level_state = {
                        'level': _loguru_logger.level(DEFAULT_GLOBAL_LEVEL)}
        return cls._instance

    def configure(self, cfg=None, log_path=None, max_bytes=None, backup_count=None,
                  file_level=None, console_level=None, global_level=None, force=False,
                  enqueue=None, intercept_stdlib=False):
        """
        这是装配日志sink的函数（幂等，线程安全）
        Args:
            cfg: 参数配置字典，键名见模块docstring
            log_path: 日志文件路径，默认./logs/yolo_deploy.log
            max_bytes: 单文件大小上限（字节），达到后轮转，默认50MB
            backup_count: 备份文件保留个数，默认7
            file_level: 文件sink级别，默认DEBUG
            console_level: 控制台sink级别，默认INFO
            global_level: 全局等级（全局等级过滤），显式参数或cfg的log_global_level键，
                          仅装配时生效，运行时调整请用set_level
            force: 是否强制重建（换路径/级别时置True）
            enqueue: 是否启用异步队列，None时读取cfg的log_enqueue键，默认True
            intercept_stdlib: 是否桥接stdlib logging，默认False
        Returns:
            self: 日志管理器单例，支持链式调用
        """
        cfg = cfg or {}
        # 解析参数，优先级：显式参数 > 配置字典 > 默认值。
        # log_path相对路径统一走resolve_config_path（绝对原样；相对CWD优先、
        # 仓库根兜底，修复：配置内./logs/xxx.log曾按进程CWD解析，非仓库根
        # 目录运行时日志散落到意外位置）
        resolved = {
            'log_path': resolve_config_path(log_path or cfg.get('log_path', DEFAULT_LOG_PATH)),
            'max_bytes': (max_bytes if max_bytes is not None
                          else cfg.get('log_max_bytes', DEFAULT_MAX_BYTES)),
            'backup_count': (backup_count if backup_count is not None
                             else cfg.get('log_backup_count', DEFAULT_BACKUP_COUNT)),
            'file_level': file_level or cfg.get('log_file_level', DEFAULT_FILE_LEVEL),
            'console_level': (console_level
                              or cfg.get('log_console_level', DEFAULT_CONSOLE_LEVEL)),
            'enqueue': cfg.get('log_enqueue', True) if enqueue is None else enqueue,
            'intercept_stdlib': intercept_stdlib,
        }
        # 双检锁：已初始化且非强制重建时直接返回，防止重复装配sink
        with self._init_lock:
            if self._file_sink_id is not None and not force:
                return self
            # 装配时按需更新全局等级（运行时调整请用set_level，无需重建sink）
            new_global_level = global_level or cfg.get('log_global_level')
            if new_global_level is not None:
                self.set_level(new_global_level)
            self._apply(**resolved)
        return self

    def reconfigure(self, **kwargs):
        """
        这是强制重建日志配置的函数
        参数语义与configure相同，未指定的项回落到cfg（如有）或默认值
        Args:
            kwargs: 与configure一致的显式参数（可含cfg）
        Returns:
            self: 日志管理器单例，支持链式调用
        """
        return self.configure(force=True, **kwargs)

    def complete(self):
        """
        这是阻塞等待异步队列全部落盘的函数
        Returns:
            self: 日志管理器单例，支持链式调用
        """
        _loguru_logger.complete()
        return self

    def shutdown(self):
        """
        这是排空队列并摘除全部sink的函数，执行后回到未初始化状态
        Returns:
            self: 日志管理器单例，支持链式调用
        """
        _loguru_logger.complete()
        # 摘除全部sink并清空状态，释放文件句柄
        with self._init_lock:
            _loguru_logger.remove()
            self._file_sink_id = None
            self._console_sink_id = None
            self._configured_path = None
        return self

    @classmethod
    def reset(cls):
        """
        这是恢复日志管理器未初始化状态的函数（单元测试隔离用）
        Returns:
            日志管理器单例
        """
        return cls().shutdown()

    def set_level(self, level):
        """
        这是动态设置全局日志等级的函数（全局等级过滤，无需重建sink，文件与控制台同时生效）
        Args:
            level: 等级名（'DEBUG'/'INFO'/'WARNING'/'ERROR'/'CRITICAL'）或loguru Level对象
        Returns:
            self: 日志管理器单例，支持链式调用
        """
        # 校验等级：字符串按名称解析为Level对象，非法名称在赋值前抛错
        if isinstance(level, str):
            level = _loguru_logger.level(level)
        # 写入可变容器，sink的filter函数在下一条日志发出时即读取新值
        self._level_state['level'] = level
        return self

    @property
    def level(self):
        """
        这是查询当前全局日志等级的函数
        Returns:
            当前全局等级（Level对象，可用.name获取名称）
        """
        return self._level_state['level']

    def _level_filter(self, record):
        """
        这是全局等级过滤器的函数，sink在每条日志发出时动态读取等级
        Args:
            record: loguru日志记录字典
        Returns:
            是否通过全局等级过滤
        """
        lv = self._level_state['level']
        return record['level'].no >= (lv.no if hasattr(lv, 'no') else lv)

    def debug(self, message, *args, **kwargs):
        """
        这是记录DEBUG等级日志的函数（委托loguru实现，支持惰性格式化）
        Args:
            message: 日志消息（支持{}占位，未过等级时不格式化）
            args: 占位符参数
            kwargs: 透传给loguru的附加参数（如feature/backtrace等）
        Returns:
        """
        _loguru_logger.debug(message, *args, **kwargs)

    def info(self, message, *args, **kwargs):
        """
        这是记录INFO等级日志的函数（委托loguru实现，支持惰性格式化）
        Args:
            message: 日志消息（支持{}占位，未过等级时不格式化）
            args: 占位符参数
            kwargs: 透传给loguru的附加参数（如feature/backtrace等）
        Returns:
        """
        _loguru_logger.info(message, *args, **kwargs)

    def warning(self, message, *args, **kwargs):
        """
        这是记录WARNING等级日志的函数（委托loguru实现，支持惰性格式化）
        Args:
            message: 日志消息（支持{}占位，未过等级时不格式化）
            args: 占位符参数
            kwargs: 透传给loguru的附加参数（如feature/backtrace等）
        Returns:
        """
        _loguru_logger.warning(message, *args, **kwargs)

    def error(self, message, *args, **kwargs):
        """
        这是记录ERROR等级日志的函数（委托loguru实现，支持惰性格式化）
        Args:
            message: 日志消息（支持{}占位，未过等级时不格式化）
            args: 占位符参数
            kwargs: 透传给loguru的附加参数（如feature/backtrace等）
        Returns:
        """
        _loguru_logger.error(message, *args, **kwargs)

    def critical(self, message, *args, **kwargs):
        """
        这是记录CRITICAL等级日志的函数（委托loguru实现，支持惰性格式化）
        Args:
            message: 日志消息（支持{}占位，未过等级时不格式化）
            args: 占位符参数
            kwargs: 透传给loguru的附加参数（如feature/backtrace等）
        Returns:
        """
        _loguru_logger.critical(message, *args, **kwargs)

    def exception(self, message, *args, **kwargs):
        """
        这是记录ERROR等级日志并附带异常堆栈的函数（应在except块内调用）
        Args:
            message: 日志消息（支持{}占位）
            args: 占位符参数
            kwargs: 透传给loguru的附加参数
        Returns:
        """
        _loguru_logger.exception(message, *args, **kwargs)

    @property
    def configured_path(self):
        """
        这是获取当前生效日志文件路径的函数
        Returns:
            日志文件路径（未初始化时为None）
        """
        return self._configured_path

    @property
    def is_configured(self):
        """
        这是查询日志管理器是否已初始化的函数
        Returns:
            是否已完成初始化
        """
        return self._file_sink_id is not None

    def _apply(self, log_path, max_bytes, backup_count, file_level, console_level,
               enqueue, intercept_stdlib):
        """
        这是按参数装配（或重建）日志sink的函数，调用方需持有_init_lock
        Args:
            log_path: 日志文件路径
            max_bytes: 单文件大小上限（字节）
            backup_count: 备份文件保留个数
            file_level: 文件sink级别
            console_level: 控制台sink级别
            enqueue: 是否启用异步队列
            intercept_stdlib: 是否桥接stdlib logging
        Returns:
        """
        # 摘除全部旧sink（含loguru自带stderr sink）
        _loguru_logger.remove()
        # 自动创建日志目录，防止目录缺失导致初始化失败
        abs_path = os.path.abspath(log_path)
        os.makedirs(os.path.dirname(abs_path), exist_ok=True)
        # 获取文件sink并设置轮转上限、保留个数、级别与格式，enqueue为多进程安全关键开关
        self._file_sink_id = _loguru_logger.add(
            abs_path,
            rotation=_bytes_to_rotation(max_bytes),
            retention=int(backup_count),
            level=file_level,                         # sink级别过滤
            filter=self._level_filter,                # 全局等级过滤，运行时动态生效
            format=FILE_FORMAT,
            encoding='utf-8',
            enqueue=enqueue,
        )
        # 获取控制台sink并设置级别与格式，直写避免与tqdm进度条错位
        self._console_sink_id = _loguru_logger.add(
            sys.stderr,
            level=console_level,
            filter=self._level_filter,                # 全局等级过滤，运行时动态生效
            format=CONSOLE_FORMAT,
            enqueue=False,
        )
        # 记录当前生效路径，供排查使用
        self._configured_path = abs_path
        # 按需桥接stdlib logging，将第三方库日志收进同一文件
        if intercept_stdlib and not self._intercept_installed:
            logging.basicConfig(handlers=[InterceptHandler()], level=0, force=True)
            self._intercept_installed = True
        # 用装配好的logger写初始化日志，验证配置已生效
        _loguru_logger.info('日志初始化完成: path={0}, rotation={1}, retention={2}'
                            .format(abs_path, _bytes_to_rotation(max_bytes), backup_count))


# 全局日志实例入口：Logger单例本身即日志门面（含全部等级函数），import即可直接记录日志
logger = Logger()


def init_logger(cfg=None, **kwargs):
    """
    这是初始化全局日志的函数（幂等，线程安全）
    Args:
        cfg: 参数配置字典
        kwargs: 显式参数（log_path/max_bytes/backup_count等），优先级高于cfg
    Returns:
        日志管理器单例（含debug/info/error等函数，可直接记录日志）
    """
    return Logger().configure(cfg=cfg, **kwargs)


def get_logger():
    """
    这是获取全局日志实例的函数
    Returns:
        logger: 全局日志实例（管理器单例，含debug/info/error等函数）
    """
    return logger


def get_configured_path():
    """
    这是获取当前生效日志文件路径的函数
    Returns:
        日志文件路径（未初始化时为None）
    """
    return Logger().configured_path


def set_global_level(level):
    """
    这是动态设置全局日志等级的函数（全局等级过滤，无需重建sink，文件与控制台同时生效）
    Args:
        level: 等级名（'DEBUG'/'INFO'/'WARNING'/'ERROR'/'CRITICAL'）或loguru Level对象
    Returns:
        日志管理器单例
    """
    return Logger().set_level(level)

def run_main():
    """
    这是主函数
    """
    # ---------- 自检 ----------
    import tempfile

    path = tempfile.mktemp(suffix='.log')

    # 自检1：单例性与幂等性。重复实例化是同一对象，重复configure不叠加sink
    mgr = Logger()
    print('自检1 单例: Logger() is Logger() → {0}'
          .format(mgr is Logger()))
    mgr.configure(log_path=path, max_bytes=50 * 1024 * 1024, backup_count=2, force=True)
    sid = mgr._file_sink_id
    mgr.configure(log_path=path, max_bytes=50 * 1024 * 1024, backup_count=2)
    print('自检1 幂等: sink id 不变 → {0}'.format(sid == mgr._file_sink_id))

    _loguru_logger.remove(mgr._console_sink_id)   # 摘掉控制台sink，避免刷屏

    # 自检2：多线程压力。8线程x500条经异步队列写文件，全部完整无撕裂
    MSG = 'Y' * 150
    def work(i):
        for j in range(500):
            mgr.info('[t{0} l{1}] {2}'.format(i, j, MSG))

    threads = [threading.Thread(target=work, args=(i,)) for i in range(8)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    mgr.complete()                                # 等待队列全部落盘
    lines = open(path, encoding='utf-8').read().splitlines()
    intact = sum(1 for ln in lines if ln.endswith(MSG))
    print('自检2 多线程: 业务日志完整行 {0}/4000, 文件总行数 {1} (含1条初始化记录)'
          .format(intact, len(lines)))

    # 自检3：多进程安全。两个子进程与主进程写同一文件，无交错无丢失
    def proc_work(i):
        for j in range(500):
            Logger().info('[p{0} l{1}] {2}'.format(i, j, MSG))

    procs = [Process(target=proc_work, args=(i,)) for i in range(2)]
    [p.start() for p in procs]
    [p.join() for p in procs]
    mgr.complete()
    lines = open(path, encoding='utf-8').read().splitlines()
    intact = sum(1 for ln in lines if ln.endswith(MSG))
    print('自检3 多进程: 业务日志完整行 {0}/5000, 文件总行数 {1}'
          .format(intact, len(lines)))

    # 自检4：轮转与保留个数。1KB上限写30条触发多次轮转，retention=2
    path2 = tempfile.mktemp(suffix='.log')
    mgr.reconfigure(log_path=path2, max_bytes=1024, backup_count=2)
    _loguru_logger.remove(mgr._console_sink_id)
    for j in range(30):
        mgr.info('[rotate {0}] {1}'.format(j, MSG))
    mgr.complete()
    # loguru轮转备份命名：<stem>.<时间戳>.<扩展名>，通配符锚在去扩展名的stem上
    rotated = [f for f in glob.glob(os.path.splitext(path2)[0] + '*') if f != path2]
    print('自检4 轮转: 主文件 {0}B, 保留备份 {1} 个: {2}'.format(
        os.path.getsize(path2), len(rotated),
        [os.path.basename(f) for f in sorted(rotated)]))

    # 自检5：生命周期。shutdown后回到未初始化状态，重新configure可再次生效
    mgr.shutdown()
    print('自检5 生命周期: shutdown 后 is_configured={0}'.format(mgr.is_configured))
    mgr.configure(log_path=path2, max_bytes=1024, backup_count=2)
    _loguru_logger.remove(mgr._console_sink_id)
    print('自检5 生命周期: 重新 configure 后 is_configured={0}, path={1}'
          .format(mgr.is_configured, mgr.configured_path == os.path.abspath(path2)))
    mgr.shutdown()

    # 自检6：全局等级动态控制。ERROR下INFO被过滤，恢复DEBUG后重新生效
    path3 = tempfile.mktemp(suffix='.log')
    mgr.configure(log_path=path3, max_bytes=50 * 1024 * 1024, backup_count=2)
    _loguru_logger.remove(mgr._console_sink_id)
    mgr.set_level('ERROR')
    for j in range(10):
        mgr.info('[blocked {0}] {1}'.format(j, MSG))
    mgr.error('[pass {0}] {1}'.format(0, MSG))
    mgr.set_level('DEBUG')
    mgr.debug('[pass {0}] {1}'.format(1, MSG))
    mgr.complete()
    lines = open(path3, encoding='utf-8').read().splitlines()
    blocked = sum(1 for ln in lines if 'blocked' in ln)
    passed = sum(1 for ln in lines if 'pass' in ln)
    print('自检6 全局等级: ERROR时INFO漏网 {0}/10 (期望0), pass落盘 {1}/2 (期望2), 总行数 {2}'
          .format(blocked, passed, len(lines)))
    mgr.shutdown()
    # 清理自检产生的临时日志文件
    for f in [path, path2, path3] + glob.glob(os.path.splitext(path2)[0] + '*'):
        if os.path.exists(f):
            os.remove(f)


if __name__ == '__main__':
    run_main()
