#!/bin/bash
# AutoDL算力平台快速部署脚本

echo "=================================="
echo "AutoDL平台部署脚本"
echo "=================================="

# AutoDL通常已经安装了conda和CUDA
echo "检测系统环境..."
echo "Python版本: $(python --version)"
echo "CUDA版本: $(nvcc --version | grep release)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# 安装依赖（AutoDL通常有基础环境）
echo ""
echo "安装项目依赖..."
bash setup.sh

# 设置HuggingFace镜像（加速国内下载）
echo ""
echo "配置HuggingFace镜像..."
export HF_ENDPOINT=https://hf-mirror.com
echo "export HF_ENDPOINT=https://hf-mirror.com" >> ~/.bashrc

echo ""
echo "=================================="
echo "✓ AutoDL部署完成！"
echo "=================================="
echo ""
echo "提示："
echo "1. 数据将下载到data/目录"
echo "2. 模型checkpoint保存到checkpoints/目录"
echo "3. 建议使用tmux防止断连：tmux new -s dream-train"
echo ""
echo "快速开始："
echo "  python scripts/download_s1k.py"
echo "  python scripts/prepare_data.py"
echo "  bash scripts/run_sft_s1k.sh 1 checkpoints/exp1"
echo "" 