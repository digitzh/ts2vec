# ts2vec

## 1 环境准备

```sh
sudo apt install nvidia-driver-535 nvidia-utils-535-server gcc
```

https://mirrors.tuna.tsinghua.edu.cn/anaconda/miniconda/

对于CUDA Version: 12.2：

```sh
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

sudo modprobe nvidia
 lsmod | grep nvidia
nvidia-smi
```

项目要求（Python 3.8、PyTorch 1.8.1）以及服务器 CUDA 版本（12.2）

```sh
pip3 install torch==1.8.1+cu111 torchvision==0.9.1+cu111 torchaudio==0.8.1 \
  -f https://mirror.sjtu.edu.cn/pytorch-wheels/torch_stable.html \
  -i https://pypi.tuna.tsinghua.edu.cn/simple

pip install -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple/

wget http://umon.api.service.ucloud.cn/static/cloudwatch/uboltagent-v1.3.0-linux-amd64.tar.gz
tar -zxf uboltagent-v1.3.0-linux-amd64.tar.gz
cd uboltagent
chmod a+x manage.sh
./manage.sh install
```

## 2 运行

### 2.1 预训练

`--feature-columns`表示只启用单一模态：
```sh
python train.py ottawa ottawa_run \
    --loader ottawa \
    --encoder-type multiscale_wavelet \
    --encoder-dropout 0.1 \
    --encoder-l2-reg 1e-4 \
    --num-scales 4 \
    --branch-channels 32 \
    --se-reduction 4 \
    --batch-size 32 \
    --lr 0.001 \
    --repr-dims 320 \
    --epochs 100 \
    --gpu 2 \
    --feature-columns accelerometer
```

### 2.2 评估

```sh
python eval.py training/ottawa__ottawa_run_20251230_234625/model.pkl \
    --eval-protocol svm \
    --dataset ottawa \
    --loader ottawa \
    --gpu 1 \
    --feature-columns accelerometer
```
