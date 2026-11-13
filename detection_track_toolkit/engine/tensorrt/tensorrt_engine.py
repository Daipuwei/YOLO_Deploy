# -*- coding: utf-8 -*-
# @Time    : 2026/9/10 08:10
# @Author  : DaiPuWei
# @Email   : daipuwei@qq.com
# @File    : tensorrt_engine.py
# @Software: PyCharm

"""
    这是定义TensorRT推理引擎的脚本。

    模块内容：
        1. TensorRTEngine：TensorRT推理引擎类，负责加载.engine/.trt模型文件
           并执行推理。对外统一契约（与ONNXRuntime引擎一致）：
           - 输入数量/元素数不符立即抛ValueError，输入按引擎binding dtype
             显式转换（fp32预处理器输出转fp16引擎输入时不再静默截断）；
             输入为一步驱动H2D直接从numpy数组取数（零CPU拷贝，实测较
             pinned中转快约1.6%），输出保留pinned缓冲D2H；
           - 推理输出为引擎原生dtype的numpy数组（fp16引擎输出fp16），
             每路输出一个数组，顺序与引擎输出一致；
           - 上下文管理：构造时为引擎创建独立CUDA上下文，推理时push/pop，
             try/finally保证异常时context栈不残留；
           - release()按序释放流/设备缓冲/执行上下文/引擎后弹出自身上下文，
             避免TRT析构期"context is destroyed"段错误；构造失败时各资源
             属性已初始化为安全空值，release()可安全调用；
           - 输入布局（NCHW/NHWC）由基类统一启发式判定（get_is_nchw）。
        2. tensorrt(...)：引擎工厂函数，注册进ENGINE_REGISTRY（注册名'tensorrt'），
           由engine.build.build_engine按需惰性导入本模块后调用；
        3. HostDeviceMem：主机/设备内存封装类。

    版本适配说明：仅支持静态shape引擎，输入输出元信息与执行API按
    TensorRT版本自动适配：
        - TensorRT 7/8：legacy binding API（engine迭代binding名/
          get_binding_shape/get_binding_dtype/binding_is_input/
          execute_async_v2(bindings=...)）；
        - TensorRT 10：新tensor API（num_io_tensors/get_tensor_name/
          get_tensor_shape/get_tensor_dtype/get_tensor_mode/
          context.set_tensor_address/execute_async_v3）。
    版本判定在构造时执行一次并缓存（self.trt_major），推理与释放路径
    按缓存分支，避免每帧hasattr探测开销。
"""

import numpy as np
import pycuda.autoinit
import pycuda.driver as cuda
import tensorrt as trt

from ..base_engine import BaseEngine
from ..base_engine import format_io_node_info
from ..build import ENGINE_REGISTRY

class HostDeviceMem(object):
    """ Host and Device Memory Package """

    def __init__(self, host_mem, device_mem):
        self.host = host_mem
        self.device = device_mem

    def __str__(self):
        return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)

    def __repr__(self):
        return self.__str__()

class TensorRTEngine(BaseEngine):

    def __init__(self, logger, tensorrt_model_path, gpu_id=0, **kwargs):
        """
        这是TensorRT模型推理引擎的初始化函数
        Args:
            logger: 日志类实例
            tensorrt_model_path: TensorRT模型文件路径
            gpu_id: gpu序号,默认为0
        """
        # 模型文件存在性检查、公共属性初始化与kwargs注入均由基类统一完成
        super(TensorRTEngine, self).__init__(logger, tensorrt_model_path, **kwargs)
        # 资源属性先行初始化为安全空值：构造中途失败时release()可安全调用，
        # 不会因属性缺失二次异常掩盖原始错误
        self.device_ctx = None
        self.engine = None
        self.context = None
        self.inputs = []
        self.outputs = []
        self.bindings = []
        self.stream = None

        # TRT版本判定（构造时执行一次并缓存，推理/释放路径按此分支）
        self.trt_major = int(trt.__version__.split('.')[0])
        self.logger.info("TensorRT版本：{0}（{1}路径）".format(trt.__version__,"legacy binding API" if self.trt_major < 10 else "新tensor API"))

        # 初始化TensorRT推理引擎相关变量
        self.device_ctx = cuda.Device(gpu_id).make_context()
        # 反序列化引擎文件（severity=ERROR：抑制TRT运行期信息日志刷屏，默认VERBOSE）
        TRT_LOGGER = trt.Logger(trt.Logger.ERROR)
        with open(self.engine_model_path, "rb") as f, \
                trt.Runtime(TRT_LOGGER) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        self.logger.info("Loaded TensorRT engine from file {}".format(self.engine_model_path))
        self.context = self.engine.create_execution_context()
        self.inputs, self.outputs, self.bindings, self.stream = self.allocate_buffers(self.context)

    def release(self):
        """
        这是释放TensorRT引擎资源的函数（覆写基类契约）
        释放顺序：同步并释放流 -> 释放设备缓冲 -> 释放执行上下文与引擎 ->
        弹出自身CUDA上下文（回到pycuda.autoinit的全局上下文）。
        必须先把自身context设为current：TRT引擎析构内部的
        ScopedCudaEvent/Stream析构依赖正确的current context，
        在错误/已销毁的context下析构会segfault。
        """
        if self.device_ctx is None:
            return
        self.device_ctx.push()
        try:
            # 同步并释放推理流
            if self.stream is not None:
                self.stream.synchronize()
                self.stream = None
            # 释放设备端缓冲（需current context）；页锁定主机内存无pycuda
            # 显式free API（2020.1），置空引用由numpy析构释放
            for buf in self.inputs + self.outputs:
                if buf.device is not None:
                    buf.device.free()
                    buf.device = None
                buf.host = None
            # 清空缓冲容器与设备地址列表
            self.inputs = []
            self.outputs = []
            self.bindings = []
            # 释放执行上下文与引擎（TRT内部资源析构依赖current context）
            self.context = None
            self.engine = None
        finally:
            # 弹出自身context，回到autoinit全局上下文
            self.device_ctx.detach()
            self.device_ctx = None

    def collect_io_infos(self):
        """
        这是按TRT版本收集输入输出节点信息的函数（名称/形状/dtype/是否输入）
        Args:
        Returns:
            io_infos: 节点信息列表，每项为(名称, 形状列表, numpy dtype, 是否输入)
        """
        io_infos = []
        if self.trt_major >= 10:
            # TensorRT 10新tensor API：按索引遍历IO张量
            for idx in range(self.engine.num_io_tensors):
                name = self.engine.get_tensor_name(idx)
                shape = list(self.engine.get_tensor_shape(name))
                dtype = trt.nptype(self.engine.get_tensor_dtype(name))
                is_input = self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT
                io_infos.append((name, shape, dtype, is_input))
        else:
            # TensorRT 7/8 legacy binding API：迭代binding名
            for binding in self.engine:
                shape = list(self.engine.get_binding_shape(binding))
                dtype = trt.nptype(self.engine.get_binding_dtype(binding))
                is_input = self.engine.binding_is_input(binding)
                io_infos.append((binding, shape, dtype, is_input))
        return io_infos

    def allocate_buffers(self, context):
        """
        这是为TensorRT推理引擎分配模型输入输出缓存的函数
        （仅支持静态shape引擎：各维必须为正整数，动态维提前明确报错）
        Args:
            context: 上下文信息
        Returns:
            inputs/outputs: HostDeviceMem列表（输入/输出）
            bindings: 设备指针列表（legacy API执行用；TRT10下为空，
                地址经context.set_tensor_address注册）
            stream: CUDA流
        """
        inputs = []
        outputs = []
        bindings = []
        stream = cuda.Stream()
        for name, shape, dtype, is_input in self.collect_io_infos():
            # 静态shape校验：各维必须为正整数（动态引擎明确报错，fail-fast）
            if any(not isinstance(dim, int) or dim <= 0 for dim in shape):
                raise ValueError("TensorRT引擎输入输出'{0}'的shape({1})含动态维，"
                                 "仅支持静态shape引擎，请固定shape后重新导出"
                                 .format(name, shape))
            size = trt.volume(shape)
            # Allocate host and device buffers
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            # 按版本注册设备地址：TRT10经set_tensor_address，7/8走bindings列表
            if self.trt_major >= 10:
                context.set_tensor_address(name, int(device_mem))
            else:
                bindings.append(int(device_mem))
            # 收集输入输出元信息（BaseEngine契约：名称/形状/类型三列表齐备），
            # 分配即逐节点打印（不另起循环），便于初始化完成后人工确认
            if is_input:
                inputs.append(HostDeviceMem(host_mem, device_mem))
                self.input_names.append(name)
                self.input_shapes.append(shape)
                self.input_types.append(np.dtype(dtype).name)
                self.logger.info(format_io_node_info("输入节点", name, shape, dtype))
            else:
                outputs.append(HostDeviceMem(host_mem, device_mem))
                self.output_names.append(name)
                self.output_shapes.append(shape)
                self.output_types.append(np.dtype(dtype).name)
                self.logger.info(format_io_node_info("输出节点", name, shape, dtype))
        return inputs, outputs, bindings, stream

    def inference(self,input_tensor):
        """
        这是TensorRT推理引擎的前向推理函数（按TRT版本选择执行API）
        Args:
            input_tensor: 输入张量列表
        Returns:
            输出张量列表（引擎原生dtype的numpy数组，顺序与引擎输出一致）
        """
        # 输入数量校验（与ONNXRuntime引擎契约一致）
        if len(input_tensor) != len(self.inputs):
            raise ValueError("输入张量数量({0})与引擎输入数量({1})不一致"
                             .format(len(input_tensor), len(self.inputs)))
        # Push to device（try/finally保证异常时context栈不残留，
        # 否则该线程后续CUDA调用全部落到错误context上）
        self.device_ctx.push()
        try:
            # 输入：一步驱动H2D直接从预处理器的numpy数组取数（零CPU拷贝）。
            # 按引擎binding dtype显式转换（fp32预处理器输出转fp16引擎输入
            # 为显式astype，不静默截断）并校验元素数；
            # 数组为局部变量且stream.synchronize()在返回前执行，生命周期安全
            for buf, tensor in zip(self.inputs, input_tensor):
                arr = np.ascontiguousarray(
                    np.asarray(tensor).astype(buf.host.dtype, copy=False))
                if arr.size != buf.host.size:
                    raise ValueError("输入张量元素数({0})与引擎输入元素数({1})不符"
                                     .format(arr.size, buf.host.size))
                cuda.memcpy_htod_async(buf.device, arr.ravel(), self.stream)
            # Run inference（按TRT版本选择执行API）
            if self.trt_major >= 10:
                self.context.execute_async_v3(stream_handle=self.stream.handle)
            else:
                self.context.execute_async_v2(bindings=self.bindings,
                                              stream_handle=self.stream.handle)
            # Transfer predictions back from the GPU.
            for out in self.outputs:
                cuda.memcpy_dtoh_async(out.host, out.device, self.stream)
            # Synchronize the stream
            self.stream.synchronize()
            results = [out.host.reshape(output_shape)
                       for out, output_shape in zip(self.outputs, self.output_shapes)]
        finally:
            # Pop the device（异常路径同样执行）
            self.device_ctx.pop()
        return results

@ENGINE_REGISTRY.register()
def tensorrt(logger, cfg, **kwargs):
    """
    这是初始化TensorRT推理引擎的函数
    Args:
        logger: 日志类实例
        cfg: 参数配置字典
        kwargs: 自定义参数字典（gpu_id缺省读cfg，其余键透传给引擎类注入为属性）
    Returns:
        TensorRT推理引擎类实例
    """
    tensorrt_model_path = cfg["engine_model_path"]
    # gpu_id必须pop：不pop会经**kwargs与位置参数重复传入（与ORT工厂语义对齐）
    gpu_id = kwargs.pop('gpu_id', cfg.get('gpu_id', 0))
    engine = TensorRTEngine(logger, tensorrt_model_path, gpu_id, **kwargs)
    return engine
