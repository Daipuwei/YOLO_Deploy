#!/bin/bash
# 脚本所在目录(用于定位 entrypoint.sh 挂载路径)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# 使用增强版getopt处理参数（支持长选项）
TEMP=$(getopt -o "" \
       --long "port:,\
               docker_container_name:,\
               docker_image_name:,\
               start_cpu_id:,\
               end_cpu_id:,\
               mount:,\
               gpu_ids:,\
               help," \
       -n "$0" -- "$@")
if [ $? != 0 ]; then
    echo "参数解析错误！使用示例：$0 --port 8080 --docker_container_name my_container --docker_image_name ubuntu:20.04 --start_cpu_id 0 --end_cpu_id 3 --gpu_ids 0,1 --mount /data:/data" >&2
    exit 1
fi
eval set -- "$TEMP"

# 解析参数
while true ; do
    case "$1" in
        --port)
            PORT=$2
            shift 2 ;;
        --docker_container_name)
            DOCKER_CONTAINER_NAME="$2"
            shift 2 ;;
        --docker_image_name)
            DOCKER_IMAGE_NAME="$2"
            shift 2 ;;
        --start_cpu_id)
            START_CPU_ID=$2
            shift 2 ;;
        --end_cpu_id)
            END_CPU_ID=$2
            shift 2 ;;
        --mount)
            MOUNT_OPTIONS=$2
            shift 2 ;;
        --gpu_ids)
            GPU_IDS=$2
            shift 2 ;;
        --help)
            echo "用法：$0 [选项]"
            echo "  --port                  容器端口（必填）"
            echo "  --docker_container_name 容器名称（必填）"
            echo "  --docker_image_name     Docker镜像名称（必填）"
            echo "  --start_cpu_id          开始CPU核心ID (选填, 默认全部核心) "
            echo "  --end_cpu_id            结束CPU核心ID (选填, 默认全部核心) "
            echo "  --mount                 挂载目录映射关系字符串(选填)"
            echo "  --gpu_ids               指定挂载的GPU编号(选填, 逗号分隔如 0,1; 默认全部)"
            exit 0 ;;
        --)
            shift; break ;;
        *)
            echo "未知选项：$1" >&2
            exit 1 ;;
    esac
done

# ===== 必填参数与数值校验 =====
if [[ -z "$PORT" || -z "$DOCKER_CONTAINER_NAME" || -z "$DOCKER_IMAGE_NAME" ]]; then
    echo "错误: --port/--docker_container_name/--docker_image_name 为必填参数" >&2
    echo "用法: $0 --port <port> --docker_container_name <name> --docker_image_name <image> [--start_cpu_id <id> --end_cpu_id <id>] [--gpu_ids 0,1] [--mount host:container,...]" >&2
    exit 1
fi
# CPU 核心范围未传时默认使用全部核心(-1)
if [[ -z "$START_CPU_ID" || -z "$END_CPU_ID" ]]; then
    START_CPU_ID=-1
    END_CPU_ID=-1
fi
if ! [[ "$PORT" =~ ^[0-9]+$ ]]; then
    echo "错误: --port 必须为正整数" >&2
    exit 1
fi
if [[ "$PORT" -lt 1 || "$PORT" -gt 65535 ]]; then
    echo "错误: --port 必须在 1-65535 范围内" >&2
    exit 1
fi
if ! [[ "$START_CPU_ID" =~ ^-?[0-9]+$ ]] || ! [[ "$END_CPU_ID" =~ ^-?[0-9]+$ ]]; then
    echo "错误: --start_cpu_id/--end_cpu_id 必须为整数(-1 表示全部核心)" >&2
    exit 1
fi

# 定义初始化 Docker 容器的函数
init_docker_container() {
   local port=$1
   local docker_container_name=$2
   local docker_image_name=$3
   local start_cpu_id=$4
   local end_cpu_id=$5
   local mount_options=$6
   local gpu_ids=$7

   echo "mount options: ${mount_options}"

   # 删除指定名称的容器
   delete_container "$docker_container_name"

   # 获取 CPU 核心绑定参数: 未指定或 -1 时使用全部核心(不设置 --cpuset-cpus)
   local cpu_ids=""
   local cpuset_option=""
   if [[ (${start_cpu_id} -eq -1) || (${end_cpu_id} -eq -1) ]]; then
      echo "使用全部 CPU 核心"
   else
      if [[ ${start_cpu_id} -gt ${end_cpu_id} ]]; then
         echo "错误: --start_cpu_id 不能大于 --end_cpu_id" >&2
         exit 1
      fi
      cpu_ids=$(get_cpu_id_string "$start_cpu_id" "$end_cpu_id")
      cpuset_option="--cpuset-cpus=$cpu_ids"
      echo "使用的 CPU 核心 ID 为: $cpu_ids"
   fi

    # 生成动态挂载参数
    local mount_options_str=""
    if [[ -z "$mount_options" ]]; then
        echo "未配置自定义挂载点"
    else
        IFS=',' read -ra mounts <<< "$mount_options"
        for mount in "${mounts[@]}"; do
            if [[ $mount != *":"* ]]; then
                echo "错误: 挂载参数 '$mount' 缺少冒号, 应为 host:container" >&2
                exit 1
            fi
            host_dir="${mount%%:*}"
            container_dir="${mount#*:}"
            if [[ -z "$host_dir" || -z "$container_dir" ]]; then
                echo "错误: 挂载参数 '$mount' 不完整, 应为 host:container" >&2
                exit 1
            fi
            mkdir -p "$host_dir"  # 自动创建宿主机目录[3](@ref)
            mount_options_str+=" -v $host_dir:$container_dir"
        done
        echo "配置了自定义挂载点: $mount_options_str"
    fi

    # GPU挂载: 指定gpu_ids则按编号挂载指定卡, 否则自动检测挂载全部
    local gpu_options_str=""
    if [[ -n "$gpu_ids" ]]; then
        if ! [[ "$gpu_ids" =~ ^[0-9]+(,[0-9]+)*$ ]]; then
            echo "错误: --gpu_ids 格式应为逗号分隔的GPU编号, 如 0,1 或 0,2" >&2
            exit 1
        fi
        # 校验GPU编号是否越界(编号必须 < 实际卡数)
        local gpu_count=0
        if command -v nvidia-smi >/dev/null 2>&1; then
            gpu_count=$(nvidia-smi -L 2>/dev/null | wc -l)
            gpu_count=$((gpu_count))  # 去掉前导空格, 转纯整数
        fi
        if [[ $gpu_count -eq 0 ]]; then
            echo "错误: 未检测到NVIDIA GPU, 无法挂载指定GPU $gpu_ids" >&2
            exit 1
        fi
        # 校验docker是否支持GPU运行时
        if ! command -v nvidia-container-runtime >/dev/null 2>&1 && ! command -v nvidia-container-toolkit >/dev/null 2>&1; then
            echo "错误: 检测到NVIDIA GPU但缺少nvidia-container-toolkit运行时, 无法挂载GPU" >&2
            exit 1
        fi
        IFS=',' read -ra gpu_id_arr <<< "$gpu_ids"
        for gid in "${gpu_id_arr[@]}"; do
            if [[ $gid -ge $gpu_count ]]; then
                echo "错误: GPU编号 $gid 越界, 本机仅 $gpu_count 张卡(编号 0-$((gpu_count-1)))" >&2
                exit 1
            fi
        done
        gpu_options_str="--gpus \"device=$gpu_ids\""
        echo "Docker容器启用GPU支持(指定GPU: $gpu_ids)"
    else
        # 自动检测: 检测到NVIDIA GPU且docker支持GPU运行时则挂载全部
        local has_gpu=0
        local has_gpu_runtime=0
        if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then
            has_gpu=1
        fi
        if command -v nvidia-container-runtime >/dev/null 2>&1 || command -v nvidia-container-toolkit >/dev/null 2>&1; then
            has_gpu_runtime=1
        fi
        if [[ $has_gpu -eq 1 && $has_gpu_runtime -eq 1 ]]; then
            gpu_options_str="--gpus all"
            echo "Docker容器启用GPU支持(检测到NVIDIA GPU及GPU运行时): $gpu_options_str"
        else
            if [[ $has_gpu -eq 1 ]]; then
                echo "Docker容器未启用GPU支持(检测到NVIDIA GPU但缺少nvidia-container-toolkit运行时)"
            else
                echo "Docker容器未启用GPU支持(未检测到NVIDIA GPU)"
            fi
        fi
    fi

#    # 判断是否启用虚拟化
#    local enable_qemu=0
#    lower_docker_image_name=$(echo "$docker_image_name" | awk '{print tolower($0)}')
#    if [[ $lower_docker_image_name == *"rk3588"* ]]; then
#       enable_qemu=1
#    fi
#
#   # 根据标志位执行命令
#   if (( enable_qemu )); then
#       echo "启动虚拟化支持"
#
#       # 注册 QEMU 仿真器[3](@ref)
#       docker run --rm --privileged multiarch/qemu-user-static:latest --reset -p yes
#
#       # 动态资源分配（物理内存的50%）[2](@ref)
#       VM_MEMORY=$(($(grep MemTotal /proc/meminfo | awk '{print $2}') / 1024 / 2))
#       VM_CPU=$(( end_cpu_id - start_cpu_id + 1 ))
#
#       # 启动 KVM 虚拟机（带后台运行）[6](@ref)
#       docker run -d --rm --privileged \
#           --device=/dev/kvm \
#           -e VM_MEMORY=$VM_MEMORY \
#           multiarch/qemu-user-static \
#           qemu-system-aarch64 -enable-kvm -cpu host -smp $VM_CPU -m $VM_MEMORY
#       sleep 3  # 等待虚拟化初始化
#   fi


   # 启动容器
   # 只放行本机连接(容器走 /tmp/.X11-unix 本地socket), 避免放开远程X访问
   xhost +local:

   # DISPLAY 为空(纯SSH/无X环境)时跳过X相关环境变量
   local display_option=""
   if [[ -n "$DISPLAY" ]]; then
       display_option="-e DISPLAY=${DISPLAY}"
   fi

   docker run -itd \
           --init \
           --ipc=host \
           --restart=always \
           --privileged=true \
           --ulimit memlock=-1 \
           --ulimit stack=67108864 \
           -p "$port:22" \
           ${cpuset_option} \
           --name "$docker_container_name" \
           --entrypoint /entrypoint.sh \
           -e LANG=zh_CN.UTF-8 \
           -e TZ="Asia/Shanghai" \
           ${display_option} \
           -e GDK_SCALE \
           -e GDK_DPI_SCALE \
           -v /etc/localtime:/etc/localtime:ro \
           -v /tmp/.X11-unix:/tmp/.X11-unix \
           -v /dev/mem:/dev/mem \
           -v /etc/locale.conf:/etc/locale.conf \
           -v /home:/home \
           -v "$SCRIPT_DIR/entrypoint.sh:/entrypoint.sh:ro" \
           ${mount_options_str} \
           ${gpu_options_str} \
           "$docker_image_name" \
           || { echo "错误: docker run 启动失败(端口被占用/镜像不存在/GPU运行时缺失等)" >&2; exit 1; }

   # 暴露端口
   expose_container_port "$docker_container_name" "$port"
}

# 获取 CPU ID 字符串数组
get_cpu_id_string() {
   local start_cpu_id=$1
   local end_cpu_id=$2
   local cpu_id_str=""

   for ((i=start_cpu_id; i<=end_cpu_id; i++)); do
       if [ -n "$cpu_id_str" ]; then
           cpu_id_str="$cpu_id_str,$i"
       else
           cpu_id_str="$i"
       fi
   done

   # 输出结果
   echo "$cpu_id_str"
}


# 定义删除指定名称容器的函数
delete_container() {
   local container_name=$1
   # 获取容器id (含已停止容器, 精确匹配容器名, 避免误删同名子串容器)
   container_id=$(docker ps -a -q --filter "name=^/${container_name}$" | head -n1)
   if [ -z "$container_id" ]; then
       echo "没有找到容器名称为 $container_name 的容器"
   else
       echo "容器名称为 $container_name 的容器 ID 是 $container_id"
       echo "删除容器名称为 $container_name 的容器"
       docker stop $container_id
       docker rm $container_id
   fi
}


# 定义校验每个容器SSH服务的函数(sshd 已由 entrypoint 前台启动, 这里只做就绪校验)
expose_container_port() {
   local container_name=$1
   local port=$2
   # 获取容器id (精确按容器名匹配)
   container_id=$(docker ps -q --filter "name=^/${container_name}$" | head -n1)
   echo "container id : $container_id"

   if [[ -z "$container_id" ]]; then
       echo "错误: 容器 $container_name 未运行, 无法校验SSH" >&2
       return 1
   fi

   # -p host_port:22 做端口转发, 容器内 sshd 由 entrypoint 以 -D 前台方式启动
   # 这里只确认 sshd 已就绪; 首次启动若需安装 openssh-server 会给足等待时间
   local ready=0
   for _ in $(seq 1 60); do
       if docker exec "$container_id" pgrep -x sshd >/dev/null 2>&1; then
           ready=1
           break
       fi
       sleep 2
   done
   if [[ $ready -eq 1 ]]; then
       echo "SSH 服务已就绪, 可通过 ssh -p ${port} root@<host> 登录"
   else
       echo "错误: 容器内 sshd 未运行(entrypoint 可能未生效)" >&2
       return 1
   fi
}

# 初始化Docker镜像
init_docker_container "$PORT" "$DOCKER_CONTAINER_NAME" "$DOCKER_IMAGE_NAME" "$START_CPU_ID" "$END_CPU_ID" "$MOUNT_OPTIONS" "$GPU_IDS"

