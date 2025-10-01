#!/bin/bash
# 使用Python venv创建虚拟环境的脚本

set -e

VENV_DIR="venv"

echo "=================================="
echo "创建Python虚拟环境"
echo "=================================="

# 检查Python版本
python_version=$(python3 --version 2>&1 | awk '{print $2}' | cut -d. -f1,2)
echo "Python版本: $python_version"

if [ $(echo "$python_version < 3.10" | bc) -eq 1 ]; then
    echo "警告: Python版本过低，推荐使用Python 3.10+"
fi

# 创建虚拟环境
echo ""
echo "创建虚拟环境到 $VENV_DIR/ ..."
python3 -m venv $VENV_DIR

# 激活虚拟环境
echo ""
echo "激活虚拟环境..."
source $VENV_DIR/bin/activate

# 运行setup.sh
echo ""
echo "安装依赖..."
bash setup.sh

echo ""
echo "=================================="
echo "✓ 虚拟环境配置完成！"
echo "=================================="
echo ""
echo "下次使用时，请先激活环境："
echo "  source $VENV_DIR/bin/activate"
echo "" 