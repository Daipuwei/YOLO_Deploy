#!/usr/bin/env bash
# -*- coding: utf-8 -*-
# @Time    : 2026/11/01
# @Author  : DaiPuWei
# @Email   : daipuwei@qq.com
# @File    : prelabel_traverse_subdirs.sh
# @Software: PyCharm

# 通用「遍历一级子目录批量预标注」脚本：对指定父目录下的每个一级子目录
# （每个子目录即一个 labelme 数据集）依次调用 prelabel_labelme_dataset.py，
# 所有预标注相关配置均可通过命令行传入。
#
# 用法：
#   bash tools/prelabel_traverse_subdirs.sh <父目录> [选项]
#
# 示例（深度检测 trt 模型）：
#   bash tools/prelabel_traverse_subdirs.sh \
#       /data/Milesight/dataset/labelme/VS125Depth/unlabel \
#       --cfg ./config/depth/yolov8.yaml \
#       --model ./model_data/yolov8s_c2f_shortcut_v18_vs125_depth_20261030_20261030_1x3x480x480_precision_0.9858_with_score_sum_singleoutput_wiouv3.trt.fp16 \
#       --gpu_id 0 --num_threads 4 \
#       --print_detection_result --export_time \
#       --skip_existing_json

set -u

# ---------- 项目根目录（脚本位于 tools/ 下，自动上溯一级） ----------
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

# ---------- 默认参数 ----------
CFG="./config/depth/yolov8.yaml"
MODEL=""
ENGINE_TYPE=""          # 空=按模型扩展名自动判定（.onnx→onnxruntime，.trt/.engine→tensorrt）
GPU_ID=0
NUM_THREADS=4
CONF=0.6
IOU=0.45
IS_NORMALIZE="True"
PRINT_FLAGS=""          # 追加 --print_detection_result / --export_time
SKIP_EXISTING_JSON=0    # 1=断点续跑（跳过已存在同名json的图像）
MERGE_FLAGS=""          # 追加 --merge_existing_json / --merge_iou_threshold
EXTRA_OPTS=""           # 额外 -o key=value 覆盖（空格分隔，透传给prelabel脚本）

usage() {
    cat <<'EOF'
用法: bash tools/prelabel_traverse_subdirs.sh <父目录> [选项]

遍历<父目录>下每个一级子目录（每个子目录=一个labelme数据集），逐个做预标注。

必填:
  <父目录>                   待遍历的父目录（其一级子目录各为一个labelme数据集）

选项:
  --cfg PATH                 配置yaml（默认 ./config/depth/yolov8.yaml）
  --model PATH               模型文件（onnx/trt），必填；引擎类型可省略、按扩展名自动判定
  --engine_type NAME         引擎类型 onnxruntime|tensorrt（默认按模型扩展名判定）
  --gpu_id N                 GPU编号（默认0）
  --num_threads N            线程数（默认4）
  --conf F                   置信度阈值（默认0.6）
  --iou F                    IoU阈值（默认0.45）
  --is_normalize BOOL        是否归一化 True|False（默认True）
  --print_detection_result   打印检测结果
  --export_time              导出耗时
  --skip_existing_json       断点续跑：跳过图像旁已存在同名json的图像
  --merge_existing_json      合并已有json标注（默认覆盖模式）
  --merge_iou_threshold F    合并模式重叠IoU阈值（默认0.7）
  -o KEY=VALUE               透传额外配置覆盖（可多次，追加到 -o 参数表）
  -h, --help                 显示本帮助
EOF
}

# 按模型扩展名判定引擎类型
detect_engine_type() {
    case "$1" in
        *.onnx) echo "onnxruntime" ;;
        *.trt|*.engine|*.trt.*) echo "tensorrt" ;;
        *) echo "onnxruntime" ;;
    esac
}

# ---------- 解析命令行 ----------
PARENT_DIR=""
while [ $# -gt 0 ]; do
    case "$1" in
        -h|--help) usage; exit 0 ;;
        --cfg) CFG="$2"; shift 2 ;;
        --model) MODEL="$2"; shift 2 ;;
        --engine_type) ENGINE_TYPE="$2"; shift 2 ;;
        --gpu_id) GPU_ID="$2"; shift 2 ;;
        --num_threads) NUM_THREADS="$2"; shift 2 ;;
        --conf) CONF="$2"; shift 2 ;;
        --iou) IOU="$2"; shift 2 ;;
        --is_normalize) IS_NORMALIZE="$2"; shift 2 ;;
        --print_detection_result) PRINT_FLAGS="$PRINT_FLAGS --print_detection_result"; shift ;;
        --export_time) PRINT_FLAGS="$PRINT_FLAGS --export_time"; shift ;;
        --skip_existing_json) SKIP_EXISTING_JSON=1; shift ;;
        --merge_existing_json) MERGE_FLAGS="$MERGE_FLAGS --merge_existing_json"; shift ;;
        --merge_iou_threshold) MERGE_FLAGS="$MERGE_FLAGS --merge_iou_threshold $2"; shift 2 ;;
        -o) EXTRA_OPTS="$EXTRA_OPTS $2"; shift 2 ;;
        --) shift; break ;;
        -*) echo "[ERROR] 未知选项: $1"; usage; exit 1 ;;
        *) if [ -z "$PARENT_DIR" ]; then PARENT_DIR="$1"; else echo "[ERROR] 多余的位置参数: $1"; usage; exit 1; fi; shift ;;
    esac
done

# ---------- 校验必填参数 ----------
if [ -z "$PARENT_DIR" ]; then
    echo "[ERROR] 缺少父目录参数"; usage; exit 1
fi
if [ -z "$MODEL" ]; then
    echo "[ERROR] 缺少模型文件，请用 --model 指定"; usage; exit 1
fi
if [ -z "$ENGINE_TYPE" ]; then
    ENGINE_TYPE="$(detect_engine_type "$MODEL")"
fi
if [ ! -d "$PARENT_DIR" ]; then
    echo "[ERROR] 父目录不存在: $PARENT_DIR"; exit 1
fi

# ---------- TensorRT 运行所需动态库路径（引擎为 tensorrt 且目录存在才追加） ----------
if [ "$ENGINE_TYPE" = "tensorrt" ] && [ -d "/opt/3rdparty/tensorrt/lib" ]; then
    export LD_LIBRARY_PATH="/opt/3rdparty/tensorrt/lib:${LD_LIBRARY_PATH:-}"
fi

# ---------- 预标注执行函数 ----------
run_prelabel() {
    local dataset_dir="$1"
    local skip_flag=""
    [ "$SKIP_EXISTING_JSON" = "1" ] && skip_flag="--skip_existing_json"
    echo ""
    echo "========== 开始预标注: ${dataset_dir} =========="
    python tools/prelabel_labelme_dataset.py \
        --cfg "$CFG" \
        --dataset_dir "$dataset_dir" \
        --gpu_id "$GPU_ID" \
        --num_threads "$NUM_THREADS" \
        $PRINT_FLAGS \
        $MERGE_FLAGS \
        $skip_flag \
        -o DetectionModel.engine_type="$ENGINE_TYPE" \
           DetectionModel.confidence_threshold="$CONF" \
           DetectionModel.iou_threshold="$IOU" \
           DetectionModel.is_normalize="$IS_NORMALIZE" \
           DetectionModel.engine_model_path="$MODEL" \
           $EXTRA_OPTS
    local rc=$?
    if [ $rc -eq 0 ]; then
        echo "[OK] 预标注完成: ${dataset_dir}"
    else
        echo "[FAIL] 预标注失败(exit=${rc}): ${dataset_dir}"
    fi
    return $rc
}

# ---------- 主循环：遍历一级子目录 ----------
FAILED=()
for dataset_dir in "$PARENT_DIR"/*/; do
    dataset_dir="${dataset_dir%/}"   # 去掉末尾斜杠
    [ -d "$dataset_dir" ] || continue
    run_prelabel "$dataset_dir" || FAILED+=("$dataset_dir")
done

echo ""
echo "================ 预标注遍历结束 ================"
if [ ${#FAILED[@]} -eq 0 ]; then
    echo "全部目录预标注成功"
else
    echo "以下目录预标注失败："
    printf '  - %s\n' "${FAILED[@]}"
fi
