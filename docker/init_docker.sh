#!/bin/bash
# 使用增强版getopt处理参数（支持长选项）
TEMP=$(getopt -o "" \
       --long "port:,\
               docker_container_name:,\
               docker_image_name:,\
               start_cpu_id:,\
               end_cpu_id:,\
               mount:,\
               help," \
       -n "$0" -- "$@")
if [ $? != 0 ]; then
    echo "参数解析错误！使用示例：$0 -p 8080 -n my_container -i ubuntu:20.04 -c 0-3" >&2
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
        --help)
            echo "用法：$0 [选项]"
            echo "  --port                  容器端口（必填）"
            echo "  --docker_container_name 容器名称（必填）"
            echo "  --docker_image_name     Docker镜像名称（必填）"
            echo "  --start_cpu_id          开始CPU核心ID (必填) "
            echo "  --end_cpu_id            结束CPU核心ID (必填) "
            echo "  --mount                 挂载目录映射关系字符串"
            exit 0 ;;
        --)
            shift; break ;;
        *)
            echo "未知选项：$1" >&2
            exit 1 ;;
    esac
done

# 定义初始化 Docker 容器的函数
init_docker_container() {
   local port=$1
   local docker_container_name=$2
   local docker_image_name=$3
   local start_cpu_id=$4
   local end_cpu_id=$5
   local mount_options=$6

   # 删除指定名称的容器
   delete_container "$docker_container_name"

   # 获取 CPU ID 字符串
   echo "$start_cpu_id,$end_cpu_id"
   if [[ (${start_cpu_id} -eq -1) || (${end_cpu_id} -eq -1) ]]; then
      start_cpu_id=0
      end_cpu_id=$(($(nproc)-1))
   fi
   cpu_ids=$(get_cpu_id_string "$start_cpu_id" "$end_cpu_id")
   echo "使用的 CPU 核心 ID 为: $cpu_ids"

    # 生成动态挂载参数
    local mount_options_str=""
    IFS=',' read -ra mounts <<< "$mount_options"
    for mount in "${mounts[@]}"; do
        if [[ $mount == *":"* ]]; then
            host_dir=$(echo $mount | cut -d':' -f1)
            container_dir=$(echo $mount | cut -d':' -f2)
            mkdir -p "$host_dir"  # 自动创建宿主机目录[3](@ref)
            mount_options_str+=" -v $host_dir:$container_dir"
        fi
    done

    # 判断是否启用虚拟化
    local enable_qemu=0
    lower_docker_image_name=$(echo "$docker_image_name" | awk '{print tolower($0)}')
    if [[ $lower_docker_image_name == *"rk3588"* ]]; then
       enable_qemu=1
    fi

   # 根据标志位执行命令
   if (( enable_qemu )); then
       echo "启动虚拟化支持"

       # 注册 QEMU 仿真器[3](@ref)
       docker run --rm --privileged multiarch/qemu-user-static:latest --reset -p yes

       # 动态资源分配（物理内存的50%）[2](@ref)
       VM_MEMORY=$(($(grep MemTotal /proc/meminfo | awk '{print $2}') / 1024 / 2))
       VM_CPU=$(( end_cpu_id - start_cpu_id + 1 ))

       # 启动 KVM 虚拟机（带后台运行）[6](@ref)
       docker run -d --rm --privileged \
           --device=/dev/kvm \
           -e VM_MEMORY=$VM_MEMORY \
           multiarch/qemu-user-static \
           qemu-system-aarch64 -enable-kvm -cpu host -smp $VM_CPU -m $VM_MEMORY
       sleep 3  # 等待虚拟化初始化
   fi


   # 启动容器
   xhost +
   docker run -itd \
           --ipc=host \
           --restart=always \
           --privileged=true \
           --ulimit memlock=-1 \
           --ulimit stack=67108864 \
           --shm-size=32g  \
           -p "$port:22" \
	         --cpuset-cpus=$cpu_ids \
	         --name $docker_container_name \
           -e LANG=zh_CN.UTF-8 \
           -e TZ="Asia/Shanghai" \
           -e DISPLAY=${DISPLAY} \
           -e GDK_SCALE \
           -e GDK_DPI_SCALE \
           -v /etc/localtime:/etc/localtime:ro \
           -v /tmp/.X11-unix:/tmp/.X11-unix \
           -v /dev/mem:/dev/mem \
		       -v /etc/locale.conf:/etc/locale.conf \
		       -v /home:/home \
		       ${mount_options_str} \
           $docker_image_name

   # 暴露端口
   expose_container_port $docker_container_name $port
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
   # 获取容器id
   container_id=`docker ps | grep $container_name | awk '{print $1}'`
   if [ -z "$container_id" ]; then
       echo "没有找到容器名称为 $container_name 的容器"
   else
       echo "容器名称为 $container_name 的容器 ID 是 $container_id"
       echo "删除容器名称为 $container_name 的容器"
       docker stop $container_id
       docker rm $container_id
   fi
}


# 定义在每个容器中ssh配置文件中暴露端口的函数
expose_container_port() {
   local container_name=$1
   local port=$2
   # 获取容器id
   container_id=`docker ps | grep $container_name | awk '{print $1}'`
   echo "container id : $container_id"

   # 将port加入ssh配置文件
   docker exec -it $container_id /bin/bash sudo echo 'Port $port' >> /etc/ssh/sshd_config
   docker exec -it $container_id /bin/bash service ssh restart
}

# 初始化Docker镜像
init_docker_container ${PORT} ${DOCKER_CONTAINER_NAME} ${DOCKER_IMAGE_NAME} ${START_CPU_ID} ${END_CPU_ID} ${MOUNT_OPTIONS}