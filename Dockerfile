# 使用官方 PyTorch 镜像（基于 CUDA 11.1 + Python 3.8）
# 此镜像已预装 CUDA 11.1 runtime、cuDNN 和 PyTorch 1.8.1，避免编译问题
FROM pytorch/pytorch:1.8.1-cuda11.1-cudnn8-runtime

# 设置环境变量，避免交互式安装提示
ENV DEBIAN_FRONTEND=noninteractive \
    TZ=Asia/Shanghai

# 安装编译依赖（用于 Bottleneck、numpy 等需构建的包）
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    nvidia-driver-535 \
    nvidia-utils-535-server \
    g++ \
    gfortran \
    libopenblas-dev \
    liblapack-dev \
    && rm -rf /var/lib/apt/lists/*

RUN modprobe nvidia

# 设置国内 pip 源以加速下载
RUN pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple

# 创建工作目录
WORKDIR /workspace

# 复制依赖文件
COPY requirements.txt .

# 安装 Python 依赖（使用国内源）
# 注意：如果 requirements.txt 中指定了 torch/torchvision，建议移除，避免冲突
RUN pip install --no-cache-dir -r requirements.txt

# 复制项目代码
COPY . .

# 可选：设置容器启动命令（根据你的实际入口脚本修改）
# CMD ["python", "your_main_script.py"]