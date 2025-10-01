#!/bin/bash
# 使用Conda创建虚拟环境的脚本

set -e

ENV_NAME="dream-sft"

echo "=================================="
echo "创建Conda环境: $ENV_NAME"
echo "=================================="

# 检查conda是否安装
if ! command -v conda &> /dev/null; then
    echo "错误: 未找到conda命令"
    echo "请先安装Anaconda或Miniconda"
    exit 1
fi

# 创建conda环境
echo ""
echo "创建Python 3.10环境..."
conda create -n $ENV_NAME python=3.10 -y

echo ""
echo "环境创建完成！"
echo ""
echo "请运行以下命令激活环境并安装依赖："
echo ""
echo "  conda activate $ENV_NAME"
echo "  bash setup.sh"
echo "" 