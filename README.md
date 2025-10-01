# Dream 7B + S1K 监督微调项目

> 🎯 使用 Dream 7B 扩散模型和 S1K 数据集进行监督微调

---

## 📖 项目介绍

这是一个使用 **Dream 7B**（扩散型大语言模型）和 **S1K 数据集**（1000个高质量推理样本）进行监督微调的项目。

- **模型**：Dream 7B (Diffusion LLM)
- **数据**：S1K-1.1 数据集
- **任务**：提升模型在推理任务上的表现
- **时间**：5天内完成

---

## 🚀 快速开始

### 第一步：查看部署指南

**[📖 DEPLOYMENT.md](DEPLOYMENT.md)** - 详细的部署说明（适用于各种算力平台）

### 第二步：环境配置（三选一）

**方式1：使用Conda（推荐）**
```bash
bash setup_conda.sh
conda activate dream-sft
bash setup.sh
```

**方式2：使用Python venv**
```bash
bash setup_venv.sh
source venv/bin/activate
```

**方式3：AutoDL平台**
```bash
bash deploy_autodl.sh
```

### 第三步：查看实施计划

所有详细的实施步骤都在 `plan/` 目录中：

- **[plan/README.md](plan/README.md)** - 📚 **从这里开始！** 文档导航和快速索引
- **[plan/00_quick_start.md](plan/00_quick_start.md)** - 🚀 快速上手指南（推荐！）
- **[plan/01_overview.md](plan/01_overview.md)** - 📋 项目概览
- **[plan/02_detailed_steps.md](plan/02_detailed_steps.md)** - 📖 详细实施步骤
- **[plan/03_technical_notes.md](plan/03_technical_notes.md)** - 🧠 技术要点分析

### 第四步：按照计划执行

按照 `plan/00_quick_start.md` 中的Day 1工作流开始！

---

## 📁 项目结构

```
dream-FineTuned/
├── README.md              # 本文件
├── plan/                  # ⭐ 实施计划（重点查看）
│   ├── README.md         # 文档导航
│   ├── 00_quick_start.md # 快速开始
│   ├── 01_overview.md    # 项目概览
│   ├── 02_detailed_steps.md # 详细步骤
│   └── 03_technical_notes.md # 技术要点
├── demand/                # 任务需求文档
│   └── demand.md         # 原始需求
├── resp/                  # 下载的仓库
│   ├── Dream/            # Dream模型仓库
│   └── s1/               # S1项目仓库
├── scripts/               # (待创建) 训练脚本
├── data/                  # (待创建) 数据目录
├── checkpoints/           # (待创建) 模型检查点
├── results/               # (待创建) 实验结果
├── notebooks/             # (待创建) Jupyter notebooks
└── docs/                  # (待创建) 实验文档
```

---

## 🎯 核心目标

1. ✅ **理解架构**：掌握Dream扩散模型的工作原理
2. ✅ **数据处理**：将S1K数据转换为Dream训练格式
3. ✅ **微调训练**：完成至少1-3个epoch的训练
4. ✅ **效果评估**：对比微调前后的性能提升
5. ✅ **文档记录**：记录过程中的问题和解决方案

---

## 📊 关键资源

### 官方资源
- [Dream GitHub](https://github.com/HKUNLP/Dream)
- [Dream Paper](https://arxiv.org/abs/2508.15487)
- [S1 GitHub](https://github.com/simplescaling/s1)
- [S1 Paper](https://arxiv.org/abs/2501.19393)
- [S1K-1.1 数据集](https://huggingface.co/datasets/simplescaling/s1K-1.1)

### 模型
- Base模型：`Dream-org/Dream-v0-Base-7B`
- Instruct模型：`Dream-org/Dream-v0-Instruct-7B`

---

## 💰 计算资源

- **预算**：100元人民币（可报销）
- **GPU需求**：
  - 最低：1x RTX 4090 (24GB)
  - 推荐：2x RTX 4090 或 1x A100 (40GB)
  - 理想：8x GPU集群

---

## ✅ 快速检查清单

在开始之前，确认：

- [ ] 已阅读 `plan/README.md`
- [ ] GPU可用且显存≥20GB
- [ ] 已安装Python 3.10和CUDA 12.1+
- [ ] 已下载Dream和S1仓库到`resp/`目录
- [ ] 理解了项目目标和时间安排

---

## 📞 获取帮助

- 📱 微信/电话：17274608033
- 📧 邮箱：info@whaletech.ai
- 🏢 地址：上海浦东张江科学之门T1模力社区6楼

---

## 🎓 面试信息

完成项目后，预约终面展示成果：
- **时间**：30分钟（15分钟展示 + 15分钟Q&A）
- **预约链接**：https://cal.com/whaletech.ai-edward.wang

---

## 📝 重要提示

1. **过程比结果重要** - 记录你的思考过程
2. **理解比调参重要** - 面试看重对架构的理解
3. **允许使用任何工具** - AI助手、搜索引擎等都可以用
4. **遇到困难及时联系** - 预算不够可以申请追加

---

## 🚀 现在开始！

```bash
# 查看详细计划
cd plan && cat README.md

# 或者直接开始
cd plan && cat 00_quick_start.md
```

**祝你成功！加油！** 💪

---

*项目创建日期：2025-10-01*
*公司：上海蓝色鲸鱼科技有限公司 🐳* # dream-fintuned
# dream-fintuned
