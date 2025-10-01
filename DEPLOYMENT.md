# 🚀 部署指南

本文档介绍如何在不同的算力平台上部署和运行本项目。

---

## 📦 文件说明

### 依赖配置文件
- `requirements.txt` - Python依赖列表
- `environment.yml` - Conda环境配置文件
- `.gitignore` - Git忽略文件列表

### 部署脚本
- `setup.sh` - 通用环境配置脚本（适用于已有Python环境）
- `setup_conda.sh` - Conda虚拟环境创建脚本
- `setup_venv.sh` - Python venv虚拟环境创建脚本
- `deploy_autodl.sh` - AutoDL平台专用部署脚本

---

## 🎯 快速部署（三种方式）

### 方式1：使用Conda（推荐）

```bash
# 步骤1：创建conda环境
bash setup_conda.sh

# 步骤2：激活环境并安装依赖
conda activate dream-sft
bash setup.sh

# 验证安装
python -c "import torch; print(torch.__version__)"
python -c "import torch; print(torch.cuda.is_available())"
```

**或者使用environment.yml一键创建：**
```bash
conda env create -f environment.yml
conda activate dream-sft
cd resp/Dream && pip install -e . && cd ../..
```

### 方式2：使用Python venv

```bash
# 一键创建并配置虚拟环境
bash setup_venv.sh

# 下次使用时激活环境
source venv/bin/activate
```

### 方式3：在现有环境中安装

```bash
# 如果已有Python 3.10环境，直接运行
bash setup.sh
```

---

## 🖥️ 各平台部署指南

### AutoDL（推荐，性价比高）

```bash
# 1. 选择镜像：PyTorch 2.x + Python 3.10
# 2. 上传代码或git clone
# 3. 运行AutoDL专用脚本
bash deploy_autodl.sh

# 4. 开始训练
python scripts/download_s1k.py
bash scripts/run_sft_s1k.sh 1 checkpoints/exp1
```

**AutoDL特别提示**：
- 已经预装conda和CUDA，无需额外安装
- 使用HF镜像加速模型下载（脚本已自动配置）
- 建议使用tmux防止断连：`tmux new -s dream-train`
- 数据保存在实例磁盘，记得定期备份到数据集盘

### 阿里云PAI / 腾讯云GPU

```bash
# 1. 选择GPU实例（建议A100或V100）
# 2. 安装conda（如果没有）
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh

# 3. 部署项目
bash setup_conda.sh
conda activate dream-sft
bash setup.sh

# 4. 开始使用
```

### 本地环境 / 实验室服务器

```bash
# 确保有CUDA环境
nvidia-smi

# 使用conda方式
bash setup_conda.sh
conda activate dream-sft
bash setup.sh

# 或使用venv方式
bash setup_venv.sh
```

### Google Colab

```python
# 在Colab notebook中运行
!git clone <your-repo-url>
%cd dream-FineTuned

# 安装依赖
!pip install torch==2.5.1 transformers==4.46.2
!pip install -r requirements.txt
!cd resp/Dream && pip install -e . && cd ../..

# 验证GPU
import torch
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))
```

---

## ⚙️ 环境验证

运行以下命令验证环境是否正确配置：

```bash
python -c "
import torch
import transformers
import datasets
print(f'✓ PyTorch版本: {torch.__version__}')
print(f'✓ Transformers版本: {transformers.__version__}')
print(f'✓ CUDA可用: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'✓ GPU型号: {torch.cuda.get_device_name(0)}')
    print(f'✓ GPU显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB')
print('✓ 环境配置正确！')
"
```

预期输出：
```
✓ PyTorch版本: 2.5.1
✓ Transformers版本: 4.46.2
✓ CUDA可用: True
✓ GPU型号: NVIDIA A100-SXM4-40GB
✓ GPU显存: 40.0 GB
✓ 环境配置正确！
```

---

## 📂 目录结构检查

配置完成后，项目应包含以下目录：

```bash
ls -la
```

应该看到：
```
drwxr-xr-x  scripts/       # 训练脚本目录
drwxr-xr-x  data/          # 数据目录
drwxr-xr-x  checkpoints/   # 模型保存目录
drwxr-xr-x  results/       # 结果目录
drwxr-xr-x  notebooks/     # Jupyter notebooks
drwxr-xr-x  docs/          # 文档目录
```

如果目录不存在，运行：
```bash
mkdir -p scripts data checkpoints results/logs results/figures notebooks docs
```

---

## 🐛 常见问题排查

### 问题1：pip install torch 很慢

**解决方案**：
```bash
# 使用清华镜像
pip install torch==2.5.1 -i https://pypi.tuna.tsinghua.edu.cn/simple

# 或使用阿里镜像
pip install torch==2.5.1 -i https://mirrors.aliyun.com/pypi/simple/
```

### 问题2：CUDA版本不匹配

**检查CUDA版本**：
```bash
nvcc --version
nvidia-smi
```

**根据CUDA版本安装对应的PyTorch**：
- CUDA 11.8: `pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu118`
- CUDA 12.1: `pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121`

### 问题3：resp/Dream目录不存在

**原因**：需要先下载Dream和S1仓库

**解决方案**：
```bash
# 克隆Dream仓库
git clone https://github.com/HKUNLP/Dream.git resp/Dream

# 克隆S1仓库
git clone https://github.com/simplescaling/s1.git resp/s1
```

### 问题4：权限问题

```bash
# 给脚本添加执行权限
chmod +x *.sh

# 或单独添加
chmod +x setup.sh setup_conda.sh setup_venv.sh deploy_autodl.sh
```

### 问题5：HuggingFace下载慢（国内）

**解决方案1：使用镜像**
```bash
export HF_ENDPOINT=https://hf-mirror.com
echo "export HF_ENDPOINT=https://hf-mirror.com" >> ~/.bashrc
```

**解决方案2：使用ModelScope**
```bash
pip install modelscope
# 在代码中使用ModelScope下载
```

---

## 🔄 更新依赖

如果需要更新项目依赖：

```bash
# 更新requirements.txt后
pip install -r requirements.txt --upgrade

# 或更新environment.yml后
conda env update -f environment.yml --prune
```

---

## 📤 导出环境

如果需要在其他机器上复现环境：

### 使用pip
```bash
pip freeze > requirements_frozen.txt
```

### 使用conda
```bash
conda env export > environment_export.yml
```

---

## 🗑️ 清理环境

### 删除conda环境
```bash
conda deactivate
conda env remove -n dream-sft
```

### 删除venv环境
```bash
deactivate
rm -rf venv/
```

---

## 💡 最佳实践

1. **使用虚拟环境**：避免污染系统Python环境
2. **使用tmux/screen**：防止SSH断连导致训练中断
3. **定期保存checkpoint**：设置合理的save_freq
4. **监控GPU使用**：使用`nvidia-smi`或`gpustat`
5. **使用wandb**：远程监控训练进度
6. **备份数据**：定期备份checkpoints和结果

---

## 📞 获取帮助

如果遇到部署问题：

1. 检查错误日志
2. 参考本文档的故障排查章节
3. 查看`plan/`目录中的详细文档
4. 联系技术支持：
   - 微信/电话：17274608033
   - 邮箱：info@whaletech.ai

---

**部署完成后，查看快速开始指南：**
```bash
cat plan/00_quick_start.md
```

祝部署顺利！🎉 