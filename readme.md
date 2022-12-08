# MMDet-YOLO代码贡献指南

### 该commit复现了plus版本的精度

### 接下来对齐打算分成以下步骤
1. 改变model和lr变化部分
2. 改变loss部分
3. 改变数据增强部分
4. 迁移到mmyolo里


## 1. 环境安装

```shell
conda create --name mmdet-yolo python=3.9 -y
conda activate mmdet-yolo

pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113

```

### 安装mmengine

```shell
git clone https://github.com/open-mmlab/mmengine
pip install -v -e mmengine/
```

### 安装mmcv dev-2.x 





```shell
git clone -b dev-2.x https://github.com/open-mmlab/mmcv.git
pip install ninja
MMCV_WITH_OPS=1 pip install -v -e .
```
#### cuda toolkit安装
如果在安装mmcv dev-2.x时报错，请安装cuda-toolkit,以下以Ubuntu20.04且pytorch cu11.3为例

[Cuda Toolkit 11.3](https://developer.nvidia.com/cuda-11.3.0-download-archive?target_os=Linux&target_arch=x86_64&Distribution=Ubuntu&target_version=20.04&target_type=runfile_local)


```shell
wget https://developer.download.nvidia.com/compute/cuda/11.3.0/local_installers/cuda_11.3.0_465.19.01_linux.run
sudo sh cuda_11.3.0_465.19.01_linux.run
# 选择不安装驱动，只安装cuda-toolkit
```

在.bashrc或者.zshrc中添加

```shell
export PATH=$PATH:/usr/local/cuda-11.3/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-11.3/lib64
```

### 安装mmdet 

```shell
git clone -b dev-3.x https://github.com/open-mmlab/mmdetection.git
pip install -v -e .
```

### 安装MMDet-YOLO

```shell
git clone https://github.com/hhaAndroid/yolobenchmark
```

# 2. 准备数据集

## mini-coco

```shell
wget https://oneflow-static.oss-cn-beijing.aliyuncs.com/datasets/minicoco.tar.gz -O data/coco.tar.gz
```

## 自定义数据集


# 3. 训练


```shell
export PYTHONPATH=./
```

```shell
# 请先把configs/yolov5/yolov5_s_16x8_300_coco_v61.py中的use_ceph设置为False然后运行以下程序
python tools/train.py configs/yolov5/yolov5_s_16x8_300_coco_v61.py
```
## 训练debug

vscode debug lauch.json的配置如下

```json
{
    "version": "0.2.0",
    "configurations": [
        {
            "name": "Debug Current Config File",
            "type": "python",
            "request": "launch",
            "program": "tools/train.py",
            "console": "integratedTerminal",
            "args": ["${file}"],
            "env": {"PYTHONPATH":"./"},
            "justMyCode": true
        }
    ]
}
```

# 4. 测试


