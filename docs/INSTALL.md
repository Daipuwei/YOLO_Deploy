# INSTALL
这是YOLO_Deploy项目中相关Python环境的安装说明。

---
# 环境安装
## ONNX环境安装
由于ONNX格式是众多推理框架的所兼容中间格式，因此首先安装ONNX相关库． 根据`requirements.txt`安装ONNX相关环境，命令如下:
```bash
conda create deploy python=3.8
conda activate deploy
pip install requirements.txt
```

## TensorRT环境安装
在[TensorRT官网](https://developer.nvidia.com/tensorrt)下载所需安装包并解压，在这里以TensorRT7.2.2.3安装举例，安装流程如下：
```bash
# 解压安装包
sudo chmod 777 /path/to/TensorRT-7.2.2.3.Ubuntu-18.04.x86_64-gnu.cuda-11.1.cudnn8.0.tar.gz
sudo tar -zxvf /path/to/TensorRT-7.2.2.3.Ubuntu-18.04.x86_64-gnu.cuda-11.1.cudnn8.0.tar.gz
sudo chmod 777 -R /path/to/TensorRT-7.2.2.3/
# 安装TensorRT
conda activate deploy
pip install cython pycuda==2019.1
cd /path/to/TensorRT-7.2.2.3/
cd python 
pip install tensorrt-7.2.2.3-cp38-none-linux_x86_64.whl
cd ..
cd graphsurgeon
pip install graphsurgeon-0.4.5-py2.py3-none-any.whl
cd ..
cd onnx_graphsurgeon
pip install onnx_graphsurgeon-0.2.6-py2.py3-none-any.whl
cd ..
cd uff
pip install uff-0.6.9-py2.py3-none-any.whl
```
安装完成后在再`~/.bashrc`中设置TensorRT的环境变量，命令如下：
```bash
sudo vim  ~/.bashrc
# 下面命令假假如~/.bashrc文件文件中，需要根据自身需要调整TensorRT文件夹的绝对路径
export TENSORRT_ROOT=/path/to/TensorRT-7.2.2.3/
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/path/to/TensorRT-7.2.2.3/lib

# 然后使配置文件生效
source ~/.bashrc
```
