# Docker 环境说明

本目录提供容器环境的初始化脚本与启动命令示例。
使用流程三步走：**拉取基础镜像 → init_docker.sh 启动容器 → SSH 连接容器**。

## 1. 目录结构

| 文件 | 说明 |
|---|---|
| `init_docker.sh` | 容器初始化脚本（通用）：解析参数 → 删除同名旧容器 → 按 CPU 核/GPU/挂载配置启动容器 → 校验 SSH 就绪 |
| `entrypoint.sh` | 容器 entrypoint（脚本自动挂载到容器 `/entrypoint.sh`）：缺则装 openssh-server、允许 root 密码登录、`sshd -D` 前台运行作为主进程 |
| `run.sh` | 历史场景的容器启动命令示例集（按需参考） |
| `README.md` | 本文档 |

## 2. 前置要求

- 宿主机已安装 Docker（≥ 19.03），当前用户有 docker 权限（`docker ps` 可执行）；
- 需要 GPU 的容器要求宿主机安装 NVIDIA 驱动与 nvidia-container-toolkit
  （`docker run --rm --gpus all ubuntu nvidia-smi` 可验证）；
- 图形界面容器（如 labelme）需宿主机有 X11 环境。

## 3. 拉取镜像

**拉取命令**

基础镜像采用英伟达官方 CUDA 镜像（Docker Hub `nvidia/cuda`）：

```bash
# 开发/部署通用镜像：CUDA 11.8 + cuDNN 8 + Ubuntu 20.04（含nvcc等开发工具）
docker pull nvidia/cuda:11.8.0-cudnn8-devel-ubuntu20.04

# 纯部署可选runtime版（体积更小，不含编译工具，按需二选一）
docker pull nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu20.04
```

**验证**

```bash
docker images | grep nvidia/cuda
# 应列出拉取的 CUDA 镜像及其 TAG
```

**失败排查**

- **网络超时/速度慢**：配置国内镜像加速器（`/etc/docker/daemon.json` 的
  `registry-mirrors`，常见如阿里云/中科大镜像源，配置后 `systemctl restart docker`）；
- **TAG 不存在**：核对 CUDA 版本/发行版/TAG 拼写，可用
  `docker search nvidia/cuda` 或 Docker Hub 页面查询可用 TAG。

> 注：不手动拉取也可以——`init_docker.sh` 启动时若本地无镜像会自动执行拉取，
> 但**手动拉取更可控**（能看到进度、可换加速源、失败信息清晰），推荐先拉后启。

## 4. 启动容器

**参数说明**

```bash
bash init_docker.sh --port <端口> \
                    --docker_container_name <容器名> \
                    --docker_image_name <镜像名> \
                    [--start_cpu_id <起始核>] \
                    [--end_cpu_id <结束核>] \
                    [--gpu_ids <GPU编号,...>] \
                    [--mount "宿主机路径:容器路径,..."] \
                    [--help]
```

| 参数 | 必填 | 说明 |
|---|---|---|
| `--port` | 是 | 容器 SSH 端口（`-p <端口>:22` 转发；容器内 sshd 由 entrypoint 前台启动，脚本启动后校验就绪，之后用 `ssh -p <端口>` 连接） |
| `--docker_container_name` | 是 | 容器名（启动前删除同名旧容器后重建，按**精确匹配** `name=^/name$` 过滤） |
| `--docker_image_name` | 是 | 镜像名（本地不存在时自动拉取） |
| `--start_cpu_id` / `--end_cpu_id` | 否 | CPU 亲和性核区间（`--cpuset-cpus`）；不传或任一传 `-1` 表示全部核 |
| `--gpu_ids` | 否 | 指定挂载的 GPU 编号（逗号分隔，如 `0,1` 或 `0,2`；挂载前校验编号越界与 nvidia-container-toolkit）；**不传则自动检测**：检测到 GPU + GPU 运行时则挂载全部，否则不挂载 |
| `--mount` | 否 | 自定义挂载，逗号分隔多组 `宿主机路径:容器路径`（宿主机目录不存在会自动创建；`/home:/home` 为固定挂载，无需在此声明） |

容器统一运行参数（脚本内置）：`--init --ipc=host --restart=always
--privileged=true --ulimit memlock=-1 --ulimit stack=67108864
--entrypoint /entrypoint.sh`，
并固定挂载 `/etc/localtime`、`/tmp/.X11-unix`、`/dev/mem`、
`/etc/locale.conf`、`/home:/home`、`entrypoint.sh:/entrypoint.sh`，
环境变量注入 `LANG/TZ/DISPLAY/GDK_SCALE/GDK_DPI_SCALE`。

**快速启动**

```bash
cd DetectionTrackToolkit-py/docker

# 本工具集的运行环境：x86 + GPU + 挂载/data与/home
bash init_docker.sh --port 44874 \
                    --docker_container_name x86_deeplearning_runtime \
                    --docker_image_name nvidia/cuda:11.8.0-cudnn8-devel-ubuntu20.04 \
                    --start_cpu_id 0 \
                    --end_cpu_id 27 \
                    --gpu_ids 0 \
                    --mount "/data:/data"
```

（`--gpu_ids` 可省略由脚本自动检测挂载全部 GPU，或按 `0,1` 逗号分隔指定多卡；
`run.sh` 中保留了历史场景的启动命令块，按需参考。）

**注意事项**

1. **不要整文件执行 `run.sh`**——它是命令示例集，整跑会按顺序重建全部 9 个容器；
2. 脚本会**先删除同名旧容器再重建**（`docker stop && docker rm`），数据请放在挂载卷中；
3. 容器名按**精确匹配**（`name=^/name$`）过滤删除，不会误删同名前缀/子串的其它容器；
4. 容器以 `--restart=always` 运行，宿主机重启后自动拉起；`--init` 使容器内 sshd 由 tini 作为 PID 1 托管，sshd 异常退出时容器自动重启；
5. 启动完成后可用 `docker ps` 确认容器状态（Up）。

## 5. 连接容器

**SSH 连接**

```bash
ssh root@127.0.0.1 -p <端口>     # 端口即启动时的 --port
```

容器内 sshd 由 entrypoint（`--entrypoint /entrypoint.sh`）以 `-D` 前台方式
启动，脚本启动后轮询 `pgrep -x sshd` 校验就绪（首次启动若需安装
openssh-server 会留足等待时间）；
连接失败时可在宿主机验证端口监听：`docker port <容器名>`（应显示 `22/tcp -> 0.0.0.0:<端口>`）。

**常用操作**

```bash
# 查看容器内挂载（宿主机/data与/home在容器内同路径可见）
df -h /data /home

# GPU 可用性（挂载了 GPU 的容器：--gpu_ids 指定卡或自动检测全部）
nvidia-smi

# 退出连接
exit
```

## 6. 搭建开发环境

容器内 Python 环境安装流程与裸机完全一致（创建虚拟环境 → 按需安装依赖 →
默认 CUDA 模式安装引擎 → 验证），统一参见 **[INSTALL.md](../docs/INSTALL.md)**，
此处不再重复。

本容器特有的两点补充：

- conda 环境 `deploy`（Python 3.8.19）已在镜像内预置，直接激活即可：

  ```bash
  source /root/miniconda3/etc/profile.d/conda.sh
  conda activate deploy
  ```

- TensorRT 库位于 `/opt/3rdparty/tensorrt/lib`，使用时设置：

  ```bash
  export LD_LIBRARY_PATH=/opt/3rdparty/tensorrt/lib:$LD_LIBRARY_PATH
  ```

## 7. 常见问题

| 现象 | 原因与处置 |
|---|---|
| docker pull 超时/极慢 | 配置国内镜像加速器（见【失败排查】） |
| ssh 连接被拒 | 容器未启动或端口未生效，`docker ps` / `docker port <容器名>` 排查 |
| 容器重启后数据丢失 | 数据未放挂载卷（`/home`、`/data` 等），容器重建只保留挂载卷内容 |

安装与推理引擎相关问题（onnxruntime CPU/GPU 互覆、Segmentation fault、
TensorRT 库找不到、`eval` 命令冲突等）参见
**[INSTALL.md](../docs/INSTALL.md) 的常见问题章节**。
