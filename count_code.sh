#!/bin/bash
# ============================================================================
# 项目文档/代码行数统计脚本
# 用法：./count_code.sh [后缀1 后缀2 ...]
#   缺省统计后缀：.py .md .yaml .sh .toml
#   示例：./count_code.sh .py .cpp .h
# 自动排除常见无关目录（__pycache__/.git/.idea/logs/result/model_data等）
# 与无关文件（*.pyc/*.log/模型二进制/图片视频等），并在结尾列出被排除的
# 目录供排查（确认没有误伤需要统计的路径）。
# ============================================================================

set -u
cd "$(dirname "$0")"

# ---------- 1) 统计后缀（命令行覆盖，缺省五种） ----------
SUFFIXES=("$@")
if [ ${#SUFFIXES[@]} -eq 0 ]; then
    SUFFIXES=(".py" ".md" ".yaml" ".sh" ".toml")
fi

# ---------- 2) 常见无关目录排除清单（可在此增删） ----------
EXCLUDE_DIR_PATTERNS=(
    ".git" ".idea" ".claude" ".vscode" ".pytest_cache" ".mypy_cache"
    "__pycache__" "*.egg-info" "node_modules" "venv" "env"
    "logs" "result" "output_videos" "model_data" "docs/1111" "docs/models"
    "docker" ".cache"
)
# ---------- 3) 常见无关文件排除清单 ----------
EXCLUDE_FILE_PATTERNS=(
    "*.pyc" "*.log" "core*" "*.bak*" "*.onnx" "*.trt" "*.engine"
    "*.pt" "*.jpg" "*.jpeg" "*.png" "*.bmp" "*.mp4" "*.avi" "*.zip" "*.tar*"
)

# ---------- 4) 构建find表达式并收集文件清单 ----------
FIND_ARGS=()
for p in "${EXCLUDE_DIR_PATTERNS[@]}"; do
    FIND_ARGS+=(-path "./$p" -prune -o)
done
FIND_ARGS+=(-type f \()
NAME_COND=""
for p in "${EXCLUDE_FILE_PATTERNS[@]}"; do
    FIND_ARGS+=(-name "$p" -prune -o)
done
SUF_COND=()
for s in "${SUFFIXES[@]}"; do
    SUF_COND+=(-o -name "*$s")
done
FIND_ARGS+=(-false "${SUF_COND[@]}" \) -print)

FILE_LIST=$(mktemp)
find . "${FIND_ARGS[@]}" 2>/dev/null | sort > "$FILE_LIST"

# ---------- 5) 按后缀统计文件数与行数 ----------
printf "%-10s %8s %12s\n" "后缀" "文件数" "总行数"
printf "%-10s %8s %12s\n" "----------" "--------" "------------"
TOTAL_FILES=0
TOTAL_LINES=0
for s in "${SUFFIXES[@]}"; do
    n=$(awk -v suf="$s" 'substr($0, length($0)-length(suf)+1)==suf' "$FILE_LIST" | wc -l)
    l=$(awk -v suf="$s" 'substr($0, length($0)-length(suf)+1)==suf' "$FILE_LIST" \
        | xargs -r wc -l | tail -1 | awk '{print $1}')
    l=${l:-0}
    printf "%-10s %8d %12d\n" "$s" "$n" "$l"
    TOTAL_FILES=$((TOTAL_FILES + n))
    TOTAL_LINES=$((TOTAL_LINES + l))
done
printf "%-10s %8s %12s\n" "----------" "--------" "------------"
printf "%-10s %8d %12d\n" "合计" "$TOTAL_FILES" "$TOTAL_LINES"

# ---------- 6) 被排除的常见无关目录排查清单 ----------
echo
echo "被排除的常见无关目录（排查用，确认无遗漏需要统计的路径）："
EXCLUDED=$(find . \( -name "__pycache__" -o -name ".git" -o -name ".idea" \
    -o -name "logs" -o -name "result" -o -name "model_data" -o -name "*.egg-info" \
    -o -name ".pytest_cache" -o -name "output_videos" -o -name "docs/1111" \
    -o -name "docker" \) -type d -prune -print 2>/dev/null | sort)
if [ -z "$EXCLUDED" ]; then
    echo "  （无）"
else
    echo "$EXCLUDED" | sed 's/^/  /'
fi

rm -f "$FILE_LIST"
