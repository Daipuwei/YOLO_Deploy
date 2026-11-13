# 各容器启动命令清单(基于新版 init_docker.sh 重写)。
#
# 相比旧版的变化:
#   1. 已移除废弃的 --use_gpu 参数。GPU 挂载改为自动检测:
#        - 宿主机有 NVIDIA GPU 且已装 nvidia-container-toolkit -> 自动挂载全部 GPU;
#        - 否则不挂载。因此下面的命令不再出现 --use_gpu。
#   2. 需要只挂部分 GPU 时, 追加 --gpu_ids 0,1 (逗号分隔指定编号); 不传 = 挂全部。
#   3. --start_cpu_id/--end_cpu_id 仍用于给容器划定 CPU 核心范围; 不传则默认全部核心。

# deeplearning_training
bash init_docker.sh --port 32868 \
                    --docker_container_name deeplearning_training \
                    --docker_image_name daipuwei/deeplearning_training:cuda11.8_ubuntu20.04 \
                    --start_cpu_id 0 \
                    --end_cpu_id 31 \
                    --mount "/store1/:/store1,/store2:/store2"

bash init_docker.sh --port 32869 \
                    --docker_container_name deeplearning_training \
                    --docker_image_name daipuwei/deeplearning_training:cuda11.8_ubuntu20.04 \
                    --start_cpu_id 0 \
                    --end_cpu_id 31 \
                    --mount "/data:/data"

bash init_docker.sh --port 32869 \
                    --docker_container_name deeplearning_training \
                    --docker_image_name daipuwei/deeplearning_training:cuda11.8_ubuntu22.04 \
                    --start_cpu_id 0 \
                    --end_cpu_id 31 \
                    --mount "/home:/home,/data:/data"

bash init_docker.sh --port 32870 \
                    --docker_container_name deeplearning_training \
                    --docker_image_name daipuwei/deeplearning_training:cuda12.6_ubuntu20.04 \
                    --start_cpu_id 0 \
                    --end_cpu_id 31 \
                    --mount "/home:/home"

# x86_onnxruntime
bash init_docker.sh --port 44874 \
                    --docker_container_name x86_deeplearning_runtime \
                    --docker_image_name daipuwei/x86_deeplearning_runtime:cuda11.8_ubuntu20.04 \
                    --start_cpu_id 0 \
                    --end_cpu_id 27 \
                    --mount "/home:/home,/data:/data"

# rv1126
bash init_docker.sh --port 44875 \
                    --docker_container_name rv1126_crosscompile_runtime \
                    --docker_image_name vot_rknn_runtime:rv1126_crosscompile \
                    --start_cpu_id 0 \
                    --end_cpu_id 19 \
                    --mount "/home:/home,/data:/data,/store1:/store1,/store2:/store2"

# rv1126b
bash init_docker.sh --port 44876 \
                    --docker_container_name rv1126b_crosscompile_runtime \
                    --docker_image_name daipuwei/rknn_runtime:rv1126b_crosscompile \
                    --start_cpu_id 0 \
                    --end_cpu_id 31 \
                    --mount "/home:/home,/data:/data,/store1:/store1,/store2:/store2"

# rk3576
bash init_docker.sh --port 44876 \
                    --docker_container_name rk3576_crosscompile_runtime_tmp \
                    --docker_image_name ubuntu:22.04 \
                    --start_cpu_id 0 \
                    --end_cpu_id 27 \
                    --mount "/home:/home,/data:/data"

bash init_docker.sh --port 44876 \
                    --docker_container_name rk3576_sdk \
                    --docker_image_name ubuntu:22.04 \
                    --start_cpu_id 0 \
                    --end_cpu_id 27 \
                    --mount "/home:/home,/data:/data"

# rknn_model_convert
bash init_docker.sh --port 44877 \
                    --docker_container_name rknn_model_convert \
                    --docker_image_name daipuwei/rknn_model_convert:rknn_toolkit1.7.5_rknn_toolkit2_2.3.2 \
                    --start_cpu_id 0 \
                    --end_cpu_id 31 \
                    --mount "/home:/home,/data:/data,/store1:/store1,/store2:/store2"

# hi3519dv500
bash init_docker.sh --port 44878 \
                    --docker_container_name hi3519dv500_crosscompile_runtime \
                    --docker_image_name daipuwei/cann_runtime:hi3519dv500_crosscompile \
                    --start_cpu_id 0 \
                    --end_cpu_id 31 \
                    --mount "/home:/home,/data:/data,/store1:/store1,/store2:/store2"

bash init_docker.sh --port 44878 \
                    --docker_container_name hi3519dv500_crosscompile_runtime \
                    --docker_image_name ubuntu:22.04 \
                    --start_cpu_id 0 \
                    --end_cpu_id 27 \
                    --mount "/home:/home,/data:/data"

# lianyong539a
bash init_docker.sh --port 32870 \
                    --docker_container_name lianyong_model_convert \
                    --docker_image_name ubuntu20p04:23071100 \
                    --start_cpu_id 0 \
                    --end_cpu_id 27 \
                    --mount "/home:/home,/data:/data"

# nt98539
bash init_docker.sh --port 32871 \
                    --docker_container_name nt98539_crosscompile \
                    --docker_image_name ubuntu:22.04 \
                    --start_cpu_id 0 \
                    --end_cpu_id 27 \
                    --mount "/home:/home,/data:/data"
