# x86_deploy
bash init_docker.sh --port 44874 \
                    --docker_container_name x86_deeplearning_runtime \
                    --docker_image_name vot_x86_deeplearning_runtime:cuda11.8_ubuntu20.04 \
                    --start_cpu_id 0 \
                    --end_cpu_id 19 \
                    --mount "/data:/data"