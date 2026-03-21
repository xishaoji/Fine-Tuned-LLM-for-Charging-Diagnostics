#!/bin/bash
echo "🚀 开始安装 Unsloth 及相关依赖..."

# 1. 更新 pip
python -m pip install --upgrade pip

# 2. 安装 Unsloth 及其强依赖 (这里针对 CUDA 12.1 和 PyTorch 2.x)
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
pip install --no-deps "xformers<0.0.27" "trl<0.9.0" peft accelerate bitsandbytes

# 3. 安装其他常用工具
pip install datasets pandas wandb

echo "✅ 环境配置完成！"