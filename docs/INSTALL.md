# INSTALL

本指南从零开始：**创建虚拟环境 → 按需安装依赖 → 准备模型配置 → 验证**。
推理引擎默认可选 **CUDA 模式**（onnxruntime-gpu），CPU 模式与 TensorRT 为按需分支。
全部命令已在 CUDA 11.8 + RTX 3060 环境实测通过。

> 若使用现成的 Docker 镜像（含 CUDA/cuDNN 预装环境），可跳过第 2 节的系统依赖
> 部分，容器启动方式见 `docker/README.md`。

## 1. 环境要求

| 项目 | 要求 |
|---|---|
| 操作系统 | Ubuntu 20.04+ |
| GPU（默认 CUDA 模式需要） | NVIDIA GPU + 驱动 ≥ 525 |
| CUDA | 11.8 |
| cuDNN | 8.x |
| Python | 3.8+（建议 3.8.19） |
| pip | ≥ 21.3（支持 PEP 660 可编辑安装） |

## 2. 从零安装流程

### 2.1 创建虚拟环境

```bash
# 方式一：conda（推荐，与已验证环境一致）
conda create -n deploy python=3.8 -y
conda activate deploy

# 方式二：venv（无conda时）
python3 -m venv deploy_env
source deploy_env/bin/activate
```

### 2.2 安装本工具集基础依赖

```bash
cd DetectionTrackToolkit-py
pip install -e . --trusted-host mirrors.aliyun.com
```

该命令以可编辑方式安装 `detection_track_toolkit` 包并自动安装基础依赖
（numpy/opencv-python/PyYAML/loguru/tqdm/tabulate/pycocotools，清单见
`pyproject.toml` 的 `dependencies`），**不安装任何推理引擎**。
可编辑安装的包指向源码目录，日常修改代码无需重新安装。

### 2.3 安装推理引擎（按需，默认 CUDA 模式）

**引擎与 onnxruntime 二选一，严禁同时安装 CPU 版与 GPU 版**
（两者共享同一目录的库文件，后装者覆盖先装者，GPU 能力会被静默遮蔽）。

**▎默认：CUDA 模式（onnxruntime-gpu）**

```bash
pip install -e ".[gpu]" --trusted-host mirrors.aliyun.com
# 等价于：pip install onnxruntime-gpu==1.18.1 --trusted-host mirrors.aliyun.com
```

验证 GPU 引擎生效：

```bash
python -c "import onnxruntime as rt; print(rt.get_available_providers())"
# 应包含 CUDAExecutionProvider
```

**▎备选一：CPU 模式（无 GPU 机器）**

```bash
pip install -e ".[cpu]" --trusted-host mirrors.aliyun.com
```

**▎备选二：TensorRT 模式（在 CUDA 模式基础上附加，追求极致吞吐）**

TensorRT 的 python 包不发布在 PyPI，需从
[TensorRT 官网](https://developer.nvidia.com/tensorrt) 下载与 CUDA 版本匹配的
安装包（本环境为 TensorRT 8.6.1 + CUDA 11.8）并解压，安装流程如下：

```bash
# 1) 解压安装包（以8.6.1为例，按实际文件名调整）
tar -zxvf /path/to/TensorRT-8.6.1.6.Linux.x86_64-gnu.cuda-11.8.tar.gz
cd /path/to/TensorRT-8.6.1.6/

# 2) 安装pycuda与TensorRT python包（wheel在解压目录内，不走pip镜像）
conda activate deploy
pip install pycuda==2020.1
pip install python/tensorrt-8.6.1-cp38-none-linux_x86_64.whl
pip install graphsurgeon/graphsurgeon-0.4.6-py2.py3-none-any.whl
pip install onnx_graphsurgeon/onnx_graphsurgeon-0.3.12-py2.py3-none-any.whl
pip install uff/uff-0.6.9-py2.py3-none-any.whl

# 3) 设置环境变量（写入~/.bashrc持久生效，按实际解压路径调整）
export TENSORRT_ROOT=/path/to/TensorRT-8.6.1.6/
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/path/to/TensorRT-8.6.1.6/lib
source ~/.bashrc

# 4) 验证
python -c "import tensorrt; print(tensorrt.__version__)"
```

本容器内 TensorRT 已按上述流程安装（库位于 `/opt/3rdparty/tensorrt/lib`），
直接使用即可。

### 2.4 环境变量与 .bashrc 陷阱

- TensorRT 库路径（仅 TensorRT 模式）：
  `export LD_LIBRARY_PATH=/opt/3rdparty/tensorrt/lib:$LD_LIBRARY_PATH`；
- 交互式终端会加载 `.bashrc`：若其中含 3rdparty 编译库路径（如
  `/opt/3rdparty/onnxruntime/lib`、`cvcuda`、`opencv` 等），其同名 `.so` 会被
  动态链接器优先加载，与 pip 版引擎 ABI 冲突导致段错误（已踩坑）。
  保持 `.bashrc` 干净，3rdparty 路径按进程在启动脚本内设置。

## 3. 模型与配置文件准备

```bash
DetectionTrackToolkit-py/
├── model_data/
│   ├── <模型文件>.onnx / .trt / .engine     # 引擎模型文件
│   └── <类别名>.txt                           # 每行一个类别名，顺序与类别id一致
├── config/
│   ├── detection.yaml                         # 主配置模板（含全部参数注释说明）
│   └── <模型>.yaml                             # 按模型类型从模板复制修改
```

模型装配要求：**单输入、静态 shape**（多输出原始模型已支持，各检测层输出
经 multioutputs_decode 融合解码；动态维度在装配阶段明确报错）；
输出格式约定见各后处理器 docstring（yolov5 融合输出 / DFL 的 4+1+nc / yolox 的 5+nc）。
预处理/后处理风格差异由配置键控制（`is_normalize`、`bbox_format`、
`output_layout`、`nms_mode`、`class_agnostic_threshold`），
语义见 `config/detection.yaml` 注释。

## 4. 验证安装

```bash
# 1) 包与注册表（应输出12个模型：ppyoloe/yolo26/yolov5~13/yolox）
python -c "from detection_track_toolkit.detection_model import MODEL_REGISTRY; print(sorted(MODEL_REGISTRY.keys()))"

# 2) 三个命令行入口
detect --help
eval_dataset --help        # 注意：命令名是eval_dataset（eval与bash内建冲突）
prelabel --help

# 3) 单元测试（无GPU/模型时部分自动跳过）
export LD_LIBRARY_PATH=/opt/3rdparty/tensorrt/lib:$LD_LIBRARY_PATH   # 有TensorRT时
python tests/test_postprocess.py
python tests/test_datasets.py
python tests/test_metrics.py
python tests/test_utils_common.py
python tests/test_eval_end2end.py
python tests/test_real_model.py    # 需要真实模型文件，缺失自动SKIP

# 4) 真实推理冒烟
detect --cfg ./config/depth/yolov5.yaml --source ./video
```

## 5. 常见问题

| 现象 | 原因与处置 |
|---|---|
| providers 无 CUDAExecutionProvider | onnxruntime 被 CPU 版覆盖，恢复口诀见 2.3 警告条 |
| 检测启动即 Segmentation fault | 交互式终端 LD_LIBRARY_PATH 被 3rdparty 库污染，检查 .bashrc；新开终端生效 |
| `import tensorrt` 报 libnvinfer.so.8 找不到 | 未设置 TensorRT 库路径，见 2.3 |
| pip 安装超时/无候选版本 | 镜像为 http 需加 `--trusted-host mirrors.aliyun.com` |
| `eval` 命令输出 bash 帮助 | `eval` 与 bash 内建同名，使用 `eval_dataset` |
| 模型装配报"输入shape各维必须为正整数" | 模型为动态 shape，需重新导出为静态 shape |

### 5.1 onnxruntime CPU/GPU 覆盖后的恢复口诀

```bash
pip uninstall -y onnxruntime onnxruntime-gpu
rm -rf $(python -c "import site; print(site.getsitepackages()[0])")/onnxruntime*
pip install onnxruntime-gpu==1.18.1 --trusted-host mirrors.aliyun.com
```

## 6. 卸载与升级

```bash
pip uninstall detection-track-toolkit                        # 卸载包（源码保留）
pip install -U -e ".[gpu]" --trusted-host mirrors.aliyun.com # 升级（变更依赖时）
```

可编辑安装（`-e`）指向源码目录，日常修改代码无需重装；仅在新增/变更依赖
或更换环境时重跑安装命令。
