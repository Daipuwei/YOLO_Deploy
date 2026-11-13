#!/bin/bash
set -e

# 容器 entrypoint: 确保 openssh-server 已装并以前台方式运行 sshd 作为主进程。
# 配合 docker run --init 使用, tini 作为 PID 1 负责信号转发与僵尸回收;
# sshd 崩了 -> 主进程退出 -> --restart=always 自动重启 -> 本脚本重新拉起 sshd。

# 1) 缺失才安装 openssh-server
if ! command -v sshd >/dev/null 2>&1; then
    echo "容器内未安装 openssh-server, 尝试自动安装..."
    ( apt-get update -qq && apt-get install -y -qq openssh-server ) \
        || yum install -y -q openssh-server \
        || { echo "错误: 自动安装 openssh-server 失败" >&2; exit 1; }
fi

# 2) 允许 root 密码登录(兼容镜像默认禁止 root 登录的情况)
if [ -f /etc/ssh/sshd_config ]; then
    sed -i "s/^[#]*PermitRootLogin.*/PermitRootLogin yes/" /etc/ssh/sshd_config
    grep -q "^PermitRootLogin" /etc/ssh/sshd_config || echo "PermitRootLogin yes" >> /etc/ssh/sshd_config
fi

# 3) 准备 sshd 运行目录
mkdir -p /run/sshd

# 4) sshd 前台运行(-D 不 daemonize), 作为容器主进程
exec "$(command -v sshd)" -D
