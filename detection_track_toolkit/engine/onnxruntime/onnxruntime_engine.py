# -*- coding: utf-8 -*-
# @Time    : 2026/9/9 06:06
# @Author  : DaiPuWei
# @Email   : daipuwei@qq.com
# @File    : onnxruntime_engine.py
# @Software: PyCharm

"""
    这是定义ONNXRuntime推理引擎的脚本。

    模块内容：
        1. ONNXRuntimeEngine：ONNXRuntime推理引擎类，负责加载模型文件并执行推理。
           版本兼容：推理路径按版本自动适配——
           >=1.8使用IO Binding+预分配输出（零拷贝，CUDA下消除逐帧分配）；
           1.2~1.7使用dict session.run兼容路径（无预分配，功能完整）；
           cudnn_conv_algo_search选项要求>=1.13（低版本自动跳过该选项）。
           对外统一契约：
           - 执行provider由引擎内部判定（CUDA可用优先，否则CPU兜底并告警），
             外部仅需传入gpu_id；
           - 输入布局（NCHW/NHWC）由引擎内部启发式判定并缓存，
             通过get_is_nchw()查询；
           - 推理输出统一为float32（float32输出零拷贝直返，fp16输出自动转换），
             每路输出一个数组，顺序与模型输出一致；
           - 输入数量不符立即抛ValueError，未知张量类型兜底float32并告警；
           - 推理采用IO Binding路径（参考OrtInfer）：
             CPU路径输入经ortvalue_from_numpy零拷贝借用numpy缓冲；
             CUDA路径输入直接绑定为设备端OrtValue；
             静态shape输出在构造时预分配OrtValue复用（消除逐帧分配/释放），
             动态shape输出逐次bind_output；
           - SessionOptions：启用全部图优化、顺序执行、intra/inter_op线程数可调；
           - CUDA provider选项：cudnn_conv_algo_search缺省HEURISTIC
             （默认EXHAUSTIVE会在首帧对conv算法做基准测试，首帧耗时可从秒级
             降到毫秒级）、arena_extend_strategy缺省kSameAsRequested、
             gpu_mem_limit_gb>0时显式设置显存上限（<=0不干预，保持ORT缺省）。
        2. onnxruntime(...)：引擎工厂函数，注册进ENGINE_REGISTRY（注册名'onnxruntime'），
           由engine.build.build_engine按需惰性导入本模块后调用；
        3. select_execution_providers(gpu_id, ...)：provider选择函数（引擎内部调用）；
        4. TENSOR_TYPE_TO_NP：ONNXRuntime张量类型字符串到numpy dtype的映射表。
"""

import numpy as np
import onnxruntime as rt

from ..base_engine import BaseEngine
from ..base_engine import format_io_node_info
from ..build import ENGINE_REGISTRY

# ============================================================================
# 引擎参数常量区：所有可调参数的缺省值集中于此，便于调试参考与统一调整
# 调整语义见各常量注释；yaml/cfg不配置这些键（工厂不读取），
# 需要调整时直接修改本区常量值即可
# ============================================================================
DEFAULT_GPU_ID = 0                          # 缺省GPU设备号（命令行--gpu_id优先）
DEFAULT_INTRA_OP_THREADS = 0                # 单算子线程数（0=不设置，ORT自动取物理核数；
                                            #   CPU推理下全核对小模型常优于小线程数）
DEFAULT_INTER_OP_THREADS = 1                # 算子间并行线程数（单流推理正确选择为1）
DEFAULT_CUDNN_CONV_ALGO_SEARCH = "HEURISTIC"  # cuDNN卷积算法选择策略，候选见下方候选元组
DEFAULT_ARENA_EXTEND_STRATEGY = "kSameAsRequested"  # CUDA显存arena扩展策略，候选见下方候选元组
DEFAULT_GPU_MEM_LIMIT_GB = 0                # 显存上限(GB)，<=0不设置（保持ORT缺省）
SESSION_LOG_SEVERITY_LEVEL = 3              # ORT自身日志级别（3=ERROR，静默运行期告警）
GRAPH_OPTIMIZATION_LEVEL = "ORT_ENABLE_ALL"  # 图优化级别（启用全部优化）
EXECUTION_MODE = "ORT_SEQUENTIAL"           # 执行模式（顺序执行，单流推理正确选择）

# 参数候选值（供校验与调试参考）
CUDNN_ALGO_SEARCH_CANDIDATES = ("HEURISTIC", "EXHAUSTIVE")
ARENA_STRATEGY_CANDIDATES = ("kNextPowerOfTwo", "kSameAsRequested")
# ============================================================================


# ONNXRuntime张量类型字符串到numpy dtype的完整映射（未知类型兜底float32并告警）
TENSOR_TYPE_TO_NP = {
    "tensor(float)": np.float32,
    "tensor(double)": np.float64,
    "tensor(float16)": np.float16,
    "tensor(int8)": np.int8,
    "tensor(uint8)": np.uint8,
    "tensor(int16)": np.int16,
    "tensor(uint16)": np.uint16,
    "tensor(int32)": np.int32,
    "tensor(uint32)": np.uint32,
    "tensor(int64)": np.int64,
    "tensor(uint64)": np.uint64,
    "tensor(bool)": np.bool_,
}


def select_execution_providers(gpu_id=0,
                               cudnn_conv_algo_search="EXHAUSTIVE",
                               arena_extend_strategy="kNextPowerOfTwo",
                               gpu_mem_limit_gb=0):
    # 签名缺省为ORT API原生默认；工具集装配入口（onnxruntime工厂）会
    # 按常量区默认值显式传入HEURISTIC/kSameAsRequested等部署取值
    """
    这是选择ONNXRuntime执行provider的函数（引擎内部调用，自行判定）
    Args:
        gpu_id: gpu设备号，缺省0（CUDA可用时写入device_id）
        cudnn_conv_algo_search: cuDNN卷积算法选择策略，缺省'HEURISTIC'
            （默认EXHAUSTIVE会在首次推理时对conv算法做穷举基准，首帧耗时
            可能达到秒级；HEURISTIC按启发式直接选取，适合部署场景）
        arena_extend_strategy: CUDA显存arena扩展策略，缺省'kSameAsRequested'
        gpu_mem_limit_gb: 显存上限（GB），<=0时不设置该选项（保持ORT缺省）
    Returns:
        provider列表：CUDA可用则优先CUDA，否则CPU兜底
    """
    # 候选值校验：非法取值立即报错并列出候选（fail-fast，调试友好）
    if cudnn_conv_algo_search not in CUDNN_ALGO_SEARCH_CANDIDATES:
        raise ValueError("未知的cudnn_conv_algo_search'{0}'，候选值：{1}"
                         .format(cudnn_conv_algo_search, CUDNN_ALGO_SEARCH_CANDIDATES))
    if arena_extend_strategy not in ARENA_STRATEGY_CANDIDATES:
        raise ValueError("未知的arena_extend_strategy'{0}'，候选值：{1}"
                         .format(arena_extend_strategy, ARENA_STRATEGY_CANDIDATES))
    available = rt.get_available_providers()
    selected = []
    if "CUDAExecutionProvider" in available:
        # CUDA额外携带device_id与性能相关选项
        cuda_opts = {
            "device_id": int(gpu_id),
            "arena_extend_strategy": arena_extend_strategy,
        }
        # 版本防御：cudnn_conv_algo_search是ORT 1.13+的选项
        # （1.13前为cudnn_conv_use_max_workspace，且ORT对未知选项
        # 会报错）；低版本跳过该选项，保持ORT默认的算法选择行为
        ort_major, ort_minor = (int(x) for x in rt.__version__.split('.')[:2])
        if (ort_major, ort_minor) >= (1, 13):
            cuda_opts["cudnn_conv_algo_search"] = cudnn_conv_algo_search
        if gpu_mem_limit_gb > 0:
            cuda_opts["gpu_mem_limit"] = int(gpu_mem_limit_gb * 1024 ** 3)
        selected.append(("CUDAExecutionProvider", cuda_opts))
    # 兜底：CUDA不可用时保留CPU，保证session始终能创建
    if "CPUExecutionProvider" not in selected:
        selected.append("CPUExecutionProvider")
    return selected


class ONNXRuntimeEngine(BaseEngine):

    def __init__(self, logger, engine_model_path, gpu_id=0,
                 intra_op_threads=0, inter_op_threads=0,
                 cudnn_conv_algo_search="EXHAUSTIVE",
                 arena_extend_strategy="kNextPowerOfTwo",
                 gpu_mem_limit_gb=0, **kwargs):
        # 签名缺省为ORT API原生默认；工具集装配入口（onnxruntime工厂）
        # 会按常量区默认值显式传入部署取值
        """
        这是ONNXRuntime推理引擎的初始化函数
        Args:
            logger: 日志类实例
            engine_model_path: 模型文件路径
            gpu_id: gpu设备号，缺省0（provider由引擎内部自行判定选择）
            intra_op_threads: 单算子并行线程数，缺省0即不设置
                （ORT自动=物理核数；实测CPU推理下4线程比全核对小模型
                反而慢约40%，故缺省交还ORT自决策，CUDA路径该参数影响甚微）
            inter_op_threads: 算子间并行线程数，缺省1（单流推理正确选择）
            cudnn_conv_algo_search: cuDNN卷积算法选择策略，缺省'HEURISTIC'
            arena_extend_strategy: CUDA显存arena扩展策略，缺省'kSameAsRequested'
            gpu_mem_limit_gb: 显存上限（GB），缺省0不干预（保持ORT缺省）
            kwargs: 其余参数透传基类注入为属性
        """
        # 模型文件存在性检查、公共属性初始化与kwargs注入均由基类统一完成
        super(ONNXRuntimeEngine, self).__init__(logger, engine_model_path, **kwargs)

        # 推理路径按版本选择（构造时判定一次并缓存）：
        # IO Binding与OrtValue均为ORT 1.8+的API，>=1.8走IO Binding+
        # 预分配输出路径；1.2~1.7走dict session.run兼容路径
        # （无预分配，功能完整，仅CUDA下缺少输出缓冲复用）
        ort_major, ort_minor = (int(x) for x in rt.__version__.split('.')[:2])
        self.use_io_binding = (ort_major, ort_minor) >= (1, 8)
        self.logger.info("推理路径：{0}".format(
            "IO Binding+预分配输出" if self.use_io_binding else "dict run兼容路径"))

        # 初始化ONNXRuntime推理引擎：provider由引擎内部判定（CUDA可用优先，否则CPU）
        self.gpu_id = int(gpu_id)
        session_option = rt.SessionOptions()
        session_option.log_severity_level = SESSION_LOG_SEVERITY_LEVEL
        session_option.graph_optimization_level = getattr(
            rt.GraphOptimizationLevel, GRAPH_OPTIMIZATION_LEVEL)
        session_option.execution_mode = getattr(rt.ExecutionMode, EXECUTION_MODE)
        if int(intra_op_threads) > 0:
            session_option.intra_op_num_threads = int(intra_op_threads)
        session_option.inter_op_num_threads = int(inter_op_threads)
        self.session = rt.InferenceSession(
            self.engine_model_path,
            sess_options=session_option,
            providers=select_execution_providers(
                self.gpu_id,cudnn_conv_algo_search,
                arena_extend_strategy,gpu_mem_limit_gb
            )
        )
        # 打印版本号（与TensorRT引擎的版本日志对齐，便于确认实际加载的ORT版本）
        self.logger.info("ONNXRuntime版本：{0}".format(rt.__version__))
        # 实际生效的provider判定并缓存（CUDA兜底失败时显式告警，避免静默跑CPU）
        self.is_cuda = self.session.get_providers()[0].startswith("CUDA")
        self.logger.info("ONNXRuntime实际生效providers：{0}"
                         .format(self.session.get_providers()))
        if not self.is_cuda:
            self.logger.warning("CUDAExecutionProvider不可用，ONNXRuntime引擎将在CPU上推理"
                                "（可用providers：{0}）".format(rt.get_available_providers()))

        # 读取输入输出节点信息（名称/形状/类型）
        for input_tensor in self.session.get_inputs():
            self.input_names.append(input_tensor.name)
            self.input_shapes.append(input_tensor.shape)
            self.input_types.append(input_tensor.type)
            # 读取即逐节点打印（不另起循环），便于初始化完成后人工确认
            _dtype = TENSOR_TYPE_TO_NP.get(input_tensor.type, np.float32)
            self.logger.info(format_io_node_info("输入节点", input_tensor.name,
                                                 input_tensor.shape, _dtype))
        for output_tensor in self.session.get_outputs():
            self.output_names.append(output_tensor.name)
            self.output_shapes.append(output_tensor.shape)
            self.output_types.append(output_tensor.type)
            _dtype = TENSOR_TYPE_TO_NP.get(output_tensor.type, np.float32)
            self.logger.info(format_io_node_info("输出节点", output_tensor.name,
                                                 output_tensor.shape, _dtype))

        # 布局判断：用基类统一启发式judge_is_nchw判定，构造时算一次并缓存，避免每次调用重算
        self.is_nchw = self.judge_is_nchw(self.input_shapes)

        # 预分配静态shape输出的OrtValue（仅各维均为正整数的输出可预分配；
        # 动态shape输出逐次bind_output分配）。预分配缓冲在推理间复用，
        # 消除CUDA路径逐帧cudaMalloc/cudaFree与CPU路径逐帧malloc的开销
        self.inputs = {}   # 输入OrtValue字典（每帧推理时重建，与outputs命名对称）
        self.outputs = {}
        # 预分配静态shape输出仅IO Binding路径有效（OrtValue为1.8+API）
        if self.use_io_binding:
            for output_name, output_shape, output_type in zip(
                    self.output_names, self.output_shapes, self.output_types):
                if all(isinstance(dim, int) and dim > 0 for dim in output_shape):
                    dtype = TENSOR_TYPE_TO_NP.get(output_type, np.float32)
                    if self.is_cuda:
                        self.outputs[output_name] = \
                            rt.OrtValue.ortvalue_from_shape_and_type(
                                output_shape, dtype, "cuda", self.gpu_id)
                    else:
                        self.outputs[output_name] = \
                            rt.OrtValue.ortvalue_from_shape_and_type(
                                output_shape, dtype)
            self.logger.info("ONNXRuntime预分配输出：{0}"
                             .format(list(self.outputs.keys())))
        self.logger.info("初始化ONNXRuntime推理引擎成功")

    def get_is_nchw(self):
        """
        这是ONNXRuntime推理引擎获取模型输入布局是否为NCHW的函数
        Returns:
            是否NCHW（构造时已缓存，由基类judge_is_nchw统一判定）
        """
        return self.is_nchw

    def release(self):
        """
        这是释放ONNXRuntime引擎资源的函数（覆写基类契约）
        onnxruntime的InferenceSession没有显式close/release API，
        其原生内存（模型权重、provider状态）由引用计数在引用归零时析构回收，
        因此释放方式就是置空session引用并清空预分配输出；释放后引擎不可再推理
        """
        self.inputs = {}
        self.outputs = {}
        self.session = None

    def inference(self, input_tensor):
        """
        这是ONNXRuntime推理引擎的前向推理函数（IO Binding路径：
        输入OrtValue零拷贝借用numpy缓冲，静态shape输出复用预分配缓冲）
        Args:
            input_tensor: 输入张量(列表)
        Returns:
            输出张量列表（每路输出一个float32数组，顺序与模型输出一致）
        """
        # 输入数量校验：少传/多传立即报错，避免zip静默截断后到session.run才爆底层错误
        if len(input_tensor) != len(self.input_names):
            raise ValueError("输入张量数量({0})与模型输入数量({1})不一致，模型输入为：{2}"
                             .format(len(input_tensor), len(self.input_names), self.input_names))
        # dict run兼容路径（ORT<1.8）：组装输入字典直接session.run，
        # 输出统一float32（与IO Binding路径契约一致）
        if not self.use_io_binding:
            input_dict = {}
            for tensor, name, ort_type in zip(input_tensor, self.input_names, self.input_types):
                dtype = TENSOR_TYPE_TO_NP.get(ort_type)
                if dtype is None:
                    self.logger.warning("未知的ONNXRuntime张量类型'{0}'，输入'{1}'按float32兜底"
                                        .format(ort_type, name))
                    dtype = np.float32
                input_dict[name] = np.asarray(tensor, dtype=dtype)
            results = self.session.run(self.output_names, input_dict)
            return [np.asarray(out, dtype=np.float32) for out in results]
        # 组装输入OrtValue：按模型声明的dtype精确转换，未知类型兜底float32并告警；
        # ascontiguousarray保证OrtValue可借用numpy缓冲（非连续切片会抛错）
        # 组装输入OrtValue：按模型声明的dtype精确转换，未知类型兜底float32并告警；
        # ascontiguousarray保证OrtValue可借用numpy缓冲（非连续切片会抛错）。
        # 输入OrtValue每帧重建存入self.inputs（CPU路径借用新numpy缓冲零拷贝，
        # CUDA路径每帧H2D），与预分配输出self.outputs命名对称
        self.inputs = {}
        for tensor, name, ort_type in zip(input_tensor, self.input_names, self.input_types):
            dtype = TENSOR_TYPE_TO_NP.get(ort_type)
            if dtype is None:
                self.logger.warning("未知的ONNXRuntime张量类型'{0}'，输入'{1}'按float32兜底"
                                    .format(ort_type, name))
                dtype = np.float32
            arr = np.ascontiguousarray(tensor, dtype=dtype)
            if self.is_cuda:
                self.inputs[name] = rt.OrtValue.ortvalue_from_numpy(
                    arr, "cuda", self.gpu_id)
            else:
                # CPU：借用numpy缓冲（零拷贝），同步推理期间缓冲生命周期安全
                self.inputs[name] = rt.OrtValue.ortvalue_from_numpy(arr)

        # 创建IO Binding并绑定输入
        io_binding = self.session.io_binding()
        for name, ort_value in self.inputs.items():
            io_binding.bind_ortvalue_input(name, ort_value)
        # 绑定输出：静态shape输出复用预分配缓冲，动态shape输出按设备分配
        for name in self.output_names:
            if name in self.outputs:
                io_binding.bind_ortvalue_output(name, self.outputs[name])
            elif self.is_cuda:
                io_binding.bind_output(name, "cuda", self.gpu_id)
            else:
                io_binding.bind_output(name)

        # 模型推理（IO Binding路径）
        self.session.run_with_iobinding(io_binding)

        # 收集输出：预分配输出从复用缓冲取数，动态输出按绑定顺序取数；
        # 输出统一为float32（后处理契约按float32计算；fp16输出在此转换）
        results = []
        for idx, name in enumerate(self.output_names):
            if name in self.outputs:
                out = self.outputs[name].numpy()
            else:
                out = io_binding.get_outputs()[idx].numpy()
            results.append(np.asarray(out, dtype=np.float32))
        return results


@ENGINE_REGISTRY.register()
def onnxruntime(logger, cfg, **kwargs):
    """
    这是初始化ONNXRuntime推理引擎的函数
    Args:
        logger: 日志类实例
        cfg: 参数配置字典（engine_model_path必填，gpu_id可选）
        kwargs: 自定义参数字典（gpu_id缺省读cfg，其余键透传给引擎类）
    Returns:
        ONNXRuntime推理引擎类实例

    引擎调优参数约定：intra_op_threads/inter_op_threads/
    cudnn_conv_algo_search/arena_extend_strategy/gpu_mem_limit_gb
    在本工厂统一初始化（缺省值取脚本常量区DEFAULT_*，调整直接改
    常量区）；ONNXRuntimeEngine与select_execution_providers的签名
    缺省保持ORT API原生默认。
    """
    gpu_id = kwargs.pop('gpu_id', cfg.get('gpu_id', DEFAULT_GPU_ID))
    # 调参变量在工厂统一初始化：缺省值取脚本常量区DEFAULT_*，
    # 需要调整时直接修改常量区（引擎/选择函数签名保持ORT API原生默认）
    intra_op_threads = DEFAULT_INTRA_OP_THREADS
    inter_op_threads = DEFAULT_INTER_OP_THREADS
    cudnn_conv_algo_search = DEFAULT_CUDNN_CONV_ALGO_SEARCH
    arena_extend_strategy = DEFAULT_ARENA_EXTEND_STRATEGY
    gpu_mem_limit_gb = DEFAULT_GPU_MEM_LIMIT_GB
    engine = ONNXRuntimeEngine(logger, cfg["engine_model_path"], gpu_id=gpu_id,
                               intra_op_threads=intra_op_threads,
                               inter_op_threads=inter_op_threads,
                               cudnn_conv_algo_search=cudnn_conv_algo_search,
                               arena_extend_strategy=arena_extend_strategy,
                               gpu_mem_limit_gb=gpu_mem_limit_gb, **kwargs)
    return engine
