#!/bin/bash
# Dream 7B + S1K 微调项目环境配置脚本
# 适用于各种算力平台的自动化部署

set -e  # 遇到错误立即退出

echo "=================================="
echo "Dream 7B + S1K 项目环境配置"
echo "=================================="

# 检查Python版本
echo "检查Python版本..."
python_version=$(python --version 2>&1 | awk '{print $2}')
echo "当前Python版本: $python_version"

# 检查CUDA
echo ""
echo "检查CUDA..."
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
    echo "✓ CUDA可用"
else
    echo "⚠ 警告: 未检测到CUDA，将使用CPU模式"
fi

# 升级pip
echo ""
echo "升级pip..."
pip install --upgrade pip

# 安装PyTorch（根据CUDA版本）
echo ""
echo "安装PyTorch 2.5.1..."
if command -v nvidia-smi &> /dev/null; then
    echo "检测到GPU，安装CUDA 12.1版本..."
    pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
else
    echo "安装CPU版本..."
    pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu
fi

# 安装核心依赖
echo ""
echo "安装核心依赖..."
pip install transformers==4.46.2

# 安装其他依赖
echo ""
echo "安装其他依赖..."
pip install -r requirements.txt

# 安装Dream包
echo ""
echo "安装Dream包..."
if [ -d "resp/Dream" ]; then
    cd resp/Dream
    pip install -e .
    cd ../..
    echo "✓ Dream包安装成功"
else
    echo "⚠ 警告: resp/Dream目录不存在，跳过Dream包安装"
fi

# 验证安装
echo ""
echo "验证环境..."
python -c "import torch; print(f'PyTorch版本: {torch.__version__}')"
python -c "import transformers; print(f'Transformers版本: {transformers.__version__}')"
python -c "import torch; print(f'CUDA可用: {torch.cuda.is_available()}')"
if python -c "import torch; exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
    python -c "import torch; print(f'GPU数量: {torch.cuda.device_count()}')"
    python -c "import torch; print(f'GPU型号: {torch.cuda.get_device_name(0)}')" 2>/dev/null || true
fi

# 创建必要的目录
echo ""
echo "创建项目目录..."
mkdir -p scripts data checkpoints results/logs results/figures notebooks docs

echo ""
echo "=================================="
echo "✓ 环境配置完成！"
echo "=================================="
echo ""
echo "下一步："
echo "1. 查看实施计划: cat plan/README.md"
echo "2. 快速开始: cat plan/00_quick_start.md"
echo "3. 下载数据: python scripts/download_s1k.py"
echo "" 