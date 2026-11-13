# 项目架构与模块调用关系

本文档归纳 detection-track-toolkit 的代码结构、模块间调用关系与解耦机制，
并给出新增模块的即插即用扩展指南（新增模块不改动现有模块的逻辑内容）。
（文档与代码同步于 2026-11-13；后续架构变更请同步更新本文档。）

## 1. 总体架构

项目采用"三块装配 + 注册器 + 配置驱动"架构：

```
config/*.yaml（配置驱动：model_type/engine_type/各阈值/布局参数）
        │
        ▼
build_model(cfg) ── MODEL_REGISTRY ──► DetectionModel（编排器，base_model.py）
        │
        ├── build_engine ─────── ENGINE_REGISTRY ──────► Engine（推理引擎，engine/）
        ├── build_preprocessor ─ PREPROCESSOR_REGISTRY ► Preprocessor（预处理，preprocess/）
        └── build_postprocessor  POSTPROCESSOR_REGISTRY ► Postprocessor（后处理，postprocess/）
        │
        ▼
detect 推理管线：
    preprocess（letterbox或resize缩放/BGR转RGB/归一化/HWC转CHW，批量；
               返回批量张量+逐图原始尺度+逐图缩放逆变换参数restore_params
               =(gain_h,gain_w,pad_h,pad_w)，即逐图w/h的缩放系数与偏移系数）
        → engine.inference（IO Binding+预分配输出）
        → 多输出：multioutputs_decode 融合为单输出等价数组（逐图像）
        → decode（两阶段低分过滤 + bbox格式转换）
        → rescale_coords（按restore_params逐轴精确还原：orig=(model-pad)/gain，
          支持top_left/center两种letterbox锚定与resize拉伸；restore_params
          缺省时兼容旧口径单一gain还原）→ nms → encode（结果字典）
```

核心思想：**每个家族（yolov5/yolox/dflyolo）的差异化逻辑全部收敛在
preprocessor/postprocessor 与模型装配类中**；通用流程（输入校验、推理调度、
过滤、NMS、坐标还原、资源释放）沉淀在基类与无状态纯函数中。

## 2. 包结构

```
DetectionTrackToolkit-py/
├── config/
│   ├── detection.yaml                  # 主配置模板（全部参数说明）
│   ├── detection/*.yaml                # 各家族模型配置模板（12个）
│   └── depth/*.yaml                    # 深度估计模型配置
├── detection_track_toolkit/            # 核心库（pip包）
│   ├── engine/                         # 推理引擎
│   │   ├── base_engine.py              #   BaseEngine：输入输出信息/布局判断/逐节点打印
│   │   ├── build.py                    #   ENGINE_REGISTRY + build_engine（惰性注册）
│   │   ├── onnxruntime/                #   ONNXRuntime引擎（IO Binding+预分配+版本双路径）
│   │   └── tensorrt/                   #   TensorRT引擎（TRT 7/8/10版本适配）
│   ├── preprocess/                     # 预处理
│   │   ├── base_preprocessor.py        #   BasePreprocessor：shape校验/批量预处理框架
│   │   ├── preprocess_utils.py         #   resize_keep_aspect_ratio/resize纯函数
│   │   ├── build.py                    #   PREPROCESSOR_REGISTRY + build_preprocessor
│   │   ├── yolov5/                     #   yolov5风格预处理（yolov5/DFL家族共用）
│   │   └── yolox/                      #   yolox风格预处理
│   ├── postprocess/                    # 后处理
│   │   ├── base_postprocessor.py       #   BasePostprocessor：管线固定/家族差异抽象
│   │   ├── postprocess_utils.py        #   纯函数：iou_xyxy/nms/rescale_coords/
│   │   │                               #   sigmoid/make_grid/softmax/dfl_decode/dist2bbox
│   │   ├── build.py                    #   POSTPROCESSOR_REGISTRY + build_postprocessor
│   │   ├── anchor_base/                #   yolov5_postprocess（anchor base家族）
│   │   └── anchor_free/                #   yolox_postprocess / dflyolo_postprocess
│   ├── detection_model/                # 检测模型装配层
│   │   ├── base_model.py               #   DetectionModel：编排器（detect/detect_video/
│   │   │                               #   计时/打印/资源释放级联）
│   │   ├── detection_utils.py          #   get_classes/get_anchors/绘制/颜色生成
│   │   ├── build.py                    #   MODEL_REGISTRY + build_model
│   │   ├── anchor_base/                #   yolov5.py（含yolov7工厂）
│   │   └── anchor_free/                #   yolox.py / dflyolo.py（yolov6~v13/yolo26/ppyoloe）
│   └── utils/
│       ├── registry_utils.py           #   Registry注册器（双模式注册/开闭原则）
│       ├── common_utils.py             #   init_config/init_logger/ArgsParser
│       ├── logger_utils.py / json_utils.py
│       ├── dataset_utils/              #   数据集加载器（base/coco/voc/yolo_v1/yolo_v2）
│       └── metric_utils/               #   指标计算（coco_metric_utils/yolo_metric_utils）
├── tools/                              # 脚本层（仅依赖detection_model与utils）
│   └── detect.py / eval.py / prelabel_labelme_dataset.py
├── tests/                              # 单元测试与真实模型端到端测试
├── docs/                               # 文档（INSTALL/QuickStart/METRICS/本文件）
└── model_data/                         # 模型文件/类别名/anchors等数据文件
```

## 3. 模块调用关系

### 3.1 装配链路（初始化时）

```
tools/*.py（detect/eval/prelabel）
    → build_model(logger, cfg["DetectionModel"], gpu_id)
        → MODEL_REGISTRY[model_type]（如"yolov5"）
            → YOLOv5.__init__
                → build_engine(cfg)                 # engine_type → ENGINE_REGISTRY
                → build_preprocessor("yolov5_preprocess", 引擎输入shape, ...)
                → get_classes/...（detection_utils读txt）
                → build_postprocessor("yolov5_postprocess", 阈值, ...)
                → DetectionModel.__init__（三块注入 + kwargs属性注入）
```

### 3.2 推理数据流（运行时）

```
image（单张/批量）
    → DetectionModel.detect
        → preprocessor.preprocess(images)          # (batch,3,H,W)张量 + 逐图像原始尺度
        #                                             + 逐图restore_params(batch,4)
        → engine.inference([image_tensor])          # 输出数组列表
        → 多输出：postprocessor.multioutputs_decode(outputs)  # 融合为逐图像单输出等价数组
        → 单输出：直接按图像切片
        → postprocessor.decode(dets)                # 两阶段过滤 + bbox转换 → [x1,y1,x2,y2,score,cls_id]
        → rescale_coords / nms（postprocess_utils纯函数；rescale按restore_params
          逐轴还原 orig=(model-pad)/gain，支持top_left/center锚定与resize拉伸）
        → encode → [{bbox, score, cls_id, cls_name}]
```

### 3.3 依赖方向规则（解耦约束）

```
tools ──► detection_model ──► engine / preprocess / postprocess ──► utils
   │           │                    │（三者互不依赖）
   └───────────┴────────────────────┴────────────────────────────► utils（无上层依赖）
```

- `utils` 不依赖任何业务模块（注册器/日志/指标/数据集纯函数）；
- `engine`/`preprocess`/`postprocess` 三者**互不引用**，只依赖 `utils`；
- `detection_model` 是唯一装配层，依赖三者与 `utils`；
- `tools` 只依赖 `detection_model` 与 `utils`，不直接触碰 engine/postprocess 内部；
- `postprocess_utils` 等工具模块为无状态纯函数，可脱离类独立调用与测试。

## 4. 解耦机制

1. **Registry 注册器**（utils/registry_utils.py）：名称→对象映射，装饰器/函数调用双模式
   注册，重名显式报错，Python -O 下同样生效；新增实现只需在模块内注册，工厂无需改动
   （开闭原则）。
2. **配置驱动装配**：`model_type`/`engine_type` 等字符串经注册表查找实例化，
   家族差异全部由配置项表达（bbox_format/output_layout/is_normalize/
   with_sigmoid/anchors_path/nms_mode 等），换模型换配置而非换代码。
3. **抽象基类契约**：BaseEngine（inference/get_input_shape/get_is_nchw/release）、
   BasePreprocessor（preprocess/preprocess_single）、BasePostprocessor
   （decode/multioutputs_decode/encode + 固定管线 postprocess）——子类只实现
   差异点，通用流程由基类统一编排。
4. **惰性注册**：engine 经 pkgutil 白名单按需 import（import engine 不拉起
   onnxruntime/tensorrt 重库）；postprocess/detection_model 在 build.py 末尾
   导入家族包触发注册（每个新家族仅追加一行导入）。
5. **无状态纯函数沉淀**：iou_xyxy/nms/rescale_coords/sigmoid/make_grid/softmax/
   dfl_decode/dist2bbox 等算法沉淀为 postprocess_utils 纯函数，类内与独立调用
   行为一致，可独立单测。
6. **资源释放契约**：preprocessor/postprocessor 基类声明空 release()，
   DetectionModel.release() 统一级联调用三块，脚本侧只调模型一次。

## 5. 扩展指南（即插即用）

以下所有扩展场景，**现有模块的逻辑内容均无需修改**；仅标注"唯一触点"处
需追加一行（注册触发约定），其余零改动。

### 5.1 新增推理引擎

1. 新建包 `detection_track_toolkit/engine/<name>/`，含 `<name>_engine.py`：
   - 继承 `BaseEngine`，实现 `inference` 等抽象方法（可选覆写 `release`）；
   - 定义工厂函数并以 `@ENGINE_REGISTRY.register()` 注册（注册名与包名一致，
     如 `onnxruntime`/`tensorrt`）。
2. 配置 `engine_type: <注册名>` 即启用。
3. **现有模块零改动**：build_engine 经 pkgutil 白名单自动发现新包并惰性导入；
   若包名与注册名不一致，在 build.py 的 `_ENGINE_MODULE_MAP` 登记（仅此例外）。

### 5.2 新增预处理风格

1. 新建包 `detection_track_toolkit/preprocess/<name>/`，含 `<name>_preprocessor.py`：
   - 继承 `BasePreprocessor`，实现 `preprocess_single`（缩放→颜色→归一化→
     (转NCHW)→连续内存化）；
   - 工厂函数 `@PREPROCESSOR_REGISTRY.register()` 注册（注册名建议
     `<风格>_preprocess`，如 `yolov5_preprocess`）。
2. 模型装配类中按注册名调用 `build_preprocessor`。
3. **唯一触点**：`preprocess/build.py` 末尾追加一行 `from . import <name>`
   （注册触发约定）。

### 5.3 新增后处理家族

1. 新建包 `detection_track_toolkit/postprocess/<anchor_base|anchor_free>/`，
   含 `<家族>_postprocess.py`：
   - 继承 `BasePostprocessor`，实现 `decode`（融合输出解码）、
     `multioutputs_decode`（多输出原始模型解码，返回逐图像融合数组列表，
     与单输出融合格式一致）、`encode`（结果字典）；
   - 工厂函数 `@POSTPROCESSOR_REGISTRY.register()` 注册（注册名建议
     `<家族>_postprocess`）。
2. 家族差异通过构造参数配置化（如 bbox_format/output_layout/with_sigmoid/
   anchors/reg_max），单输出与多输出走同一 decode 管线。
3. **唯一触点**：`postprocess/build.py` 末尾追加一行导入；`postprocess/__init__.py`
   按需补导出。

### 5.4 新增检测模型（新家族）

1. 新建 `detection_track_toolkit/detection_model/<anchor_base|anchor_free>/<家族>.py`：
   - 继承 `DetectionModel`，在 `__init__` 内完成三块装配
     （build_engine → build_preprocessor → build_postprocessor → super().__init__）；
   - 工厂函数 `@MODEL_REGISTRY.register()` 注册（注册名即 model_type，如 `yolov5`）。
2. 新增配置模板 `config/detection/<家族>.yaml`（主配置中的参数说明同步一节）。
3. **唯一触点**：`detection_model/build.py` 末尾追加一行导入。

### 5.5 新增共享工具函数

- 算法类纯函数（IoU/NMS/解码/坐标还原等）追加到 `postprocess_utils.py`
  （无状态、可独立调用）；
- 通用读取/绘制工具追加到 `detection_utils.py`；
- **零触点**：纯追加，不改任何既有函数。

### 5.6 新增数据集/指标

- 数据集加载器继承 `dataset_utils/base_dataset.py`，在 `dataset_utils/build.py`
  按既有模式注册；指标函数追加到 `metric_utils/`。

## 6. 新增模块约定清单

| 项 | 约定 |
|----|------|
| 命名 | 函数/变量名首字符不使用下划线；注册名全小写、家族后缀统一（`_postprocess`） |
| 工厂 | 每模块一个工厂函数，签名与既有工厂对齐（logger/cfg/**kwargs 或位置参数+**kwargs） |
| 参数 | 家族差异参数化（配置驱动），不为单一模型写死；缺省值与基类/API原生缺省一致 |
| 注释 | docstring 采用"这是…的函数/类"句式 + Args/Returns 分节；行内注释说明形状与公式 |
| 多输出 | `multioutputs_decode` 返回逐图像融合数组列表，列布局与单输出融合格式一致，复用 `decode` |
| 过滤语义 | 两阶段低分过滤（先家族前置分、后类别最大分，同一阈值）；bbox 格式转换放在 mask 之后 |
| 资源 | 原生资源在 `release()` 释放（置空引用/清空缓冲），脚本统一经模型侧调用 |
| 测试 | 纯函数与后处理解码用例进 `tests/`（手算期望值），真实模型端到端进 `test_real_model.py` |
| 配置 | 新参数同步主配置 `config/detection.yaml` 与对应家族模板，注释写明缺省与语义 |

