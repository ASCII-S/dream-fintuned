# 🚀 快速开始指南

> **目标**: 最快上手，立即开始实验！

---

## ⏱️ 5分钟快速开始

如果你想立即开始实验，跟着这个清单走：

### ✅ 检查清单

```bash
# 1. 验证环境
□ GPU可用且显存≥20GB
□ Python 3.10
□ CUDA 12.1+

# 2. 安装依赖 (10分钟)
□ pip install torch==2.5.1 transformers==4.46.2
□ cd resp/Dream && pip install -e .

# 3. 测试推理 (5分钟)
□ 运行 test_inference.py 确认模型能加载

# 4. 准备数据 (15分钟)
□ 运行 download_s1k.py 下载数据
□ 运行 prepare_data_for_dream.py 转换格式

# 5. 开始训练！
□ 运行 bash scripts/run_sft_s1k.sh 1 checkpoints/exp1
```

---

## 📁 推荐的目录结构

创建以下目录结构，让工作更有条理：

```
dream-FineTuned/
├── resp/                      # 已有：下载的仓库
│   ├── Dream/
│   └── s1/
├── plan/                      # 已有：实施计划（当前文档）
├── scripts/                   # 创建：训练脚本
│   ├── run_sft_s1k.sh
│   ├── test_inference.py
│   └── evaluate.py
├── data/                      # 创建：数据目录
│   ├── s1k/                   # 原始S1K数据
│   └── s1k_dream_format/      # 转换后的数据
├── checkpoints/               # 创建：模型检查点
│   └── exp1/
├── results/                   # 创建：实验结果
│   ├── logs/
│   ├── figures/
│   └── eval_results.json
├── notebooks/                 # 创建：Jupyter notebooks
│   ├── explore_data.ipynb
│   └── analyze_results.ipynb
└── docs/                      # 创建：文档和笔记
    ├── architecture_notes.md
    ├── training_log.md
    └── final_report.md
```

---

## 🛠️ 第一天工作流（Day 1）

### 上午：环境准备 (3-4小时)

#### 步骤1：创建工作目录
```bash
cd /workplace/home/sunzhongao/gpu-usage/dream-FineTuned
mkdir -p scripts data checkpoints results notebooks docs
```

#### 步骤2：安装依赖
```bash
# 创建conda环境
conda create -n dream-sft python=3.10 -y
conda activate dream-sft

# 安装核心库
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
pip install transformers==4.46.2 datasets accelerate
pip install wandb tensorboard tqdm pandas pyarrow

# 安装Dream
cd resp/Dream
pip install -e .
cd ../..
```

#### 步骤3：测试环境
```bash
# 创建测试脚本
cat > scripts/test_env.py << 'EOF'
import torch
import transformers
print(f"PyTorch: {torch.__version__}")
print(f"Transformers: {transformers.__version__}")
print(f"CUDA: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
EOF

python scripts/test_env.py
```

### 下午：数据准备 (3-4小时)

#### 步骤4：下载S1K数据
```bash
cat > scripts/download_s1k.py << 'EOF'
from datasets import load_dataset
import os

print("下载S1K-1.1数据集...")
dataset = load_dataset("simplescaling/s1K-1.1", split="train")
print(f"数据集大小: {len(dataset)}")

save_dir = "data/s1k"
os.makedirs(save_dir, exist_ok=True)
dataset.save_to_disk(save_dir)
print(f"保存到: {save_dir}")

# 打印样本
print("\n第一个样本:")
print(dataset[0])
EOF

python scripts/download_s1k.py
```

#### 步骤5：转换数据格式
```bash
cat > scripts/prepare_data.py << 'EOF'
from datasets import load_from_disk, Dataset
import os

def convert_format(sample):
    question = sample.get('question', sample.get('prompt', ''))
    answer = sample.get('answer', sample.get('response', ''))
    
    prompt = [
        {"role": "system", "content": "You are a helpful AI assistant. Think step by step."},
        {"role": "user", "content": question}
    ]
    
    return {"prompt": prompt, "response": answer}

# 加载数据
dataset = load_from_disk("data/s1k")
print(f"原始数据: {len(dataset)}")

# 转换
converted = [convert_format(s) for s in dataset]
converted_dataset = Dataset.from_list(converted)

# 划分训练/验证集
split = converted_dataset.train_test_split(test_size=0.1, seed=42)

# 保存
output_dir = "data/s1k_dream_format"
os.makedirs(output_dir, exist_ok=True)
split['train'].to_parquet(f"{output_dir}/train.parquet")
split['test'].to_parquet(f"{output_dir}/val.parquet")

print(f"训练集: {len(split['train'])}")
print(f"验证集: {len(split['test'])}")
print(f"保存到: {output_dir}")
EOF

python scripts/prepare_data.py
```

#### 步骤6：验证数据
```bash
cat > scripts/validate_data.py << 'EOF'
import pandas as pd
from transformers import AutoTokenizer

train_df = pd.read_parquet("data/s1k_dream_format/train.parquet")
print(f"训练样本数: {len(train_df)}")

tokenizer = AutoTokenizer.from_pretrained("Dream-org/Dream-v0-Base-7B", trust_remote_code=True)

lengths = []
for _, row in train_df.iterrows():
    messages = row['prompt'] + [{"role": "assistant", "content": row['response']}]
    tokens = tokenizer.apply_chat_template(messages, tokenize=True)
    lengths.append(len(tokens))

import numpy as np
print(f"Token长度: min={min(lengths)}, max={max(lengths)}, mean={np.mean(lengths):.0f}")
print(f"超过2048的样本: {sum(1 for l in lengths if l > 2048)}")
EOF

python scripts/validate_data.py
```

---

## 🏃 第二天工作流（Day 2-3）：开始训练

### 创建训练脚本

```bash
cat > scripts/run_sft_s1k.sh << 'EOF'
#!/bin/bash
set -x

nproc_per_node=${1:-1}
save_path=${2:-checkpoints/s1k-sft}

cd resp/Dream

torchrun --standalone --nnodes=1 --nproc_per_node=$nproc_per_node \
    -m src.trainer.fsdp_sft_trainer \
    diffusion.time_reweighting=cart \
    data.train_files=../../data/s1k_dream_format/train.parquet \
    data.val_files=../../data/s1k_dream_format/val.parquet \
    data.max_length=2048 \
    data.prompt_key=prompt \
    data.response_key=response \
    data.truncation=right \
    optim.lr=2e-6 \
    data.micro_batch_size_per_gpu=2 \
    data.enable_perbatch_cutoff=True \
    data.perbatch_cutoff_type=random_with_input_pad \
    model.partial_pretrain=Dream-org/Dream-v0-Base-7B \
    model.trust_remote_code=True \
    model.enable_gradient_checkpointing=True \
    trainer.default_local_dir=../../$save_path \
    trainer.project_name=dream-s1k-sft \
    trainer.experiment_name=exp-$(date +%Y%m%d-%H%M) \
    trainer.logger=['console','wandb'] \
    trainer.total_epochs=3 \
    trainer.save_freq=500
EOF

chmod +x scripts/run_sft_s1k.sh
```

### 启动训练

```bash
# 单GPU训练（24GB显存）
bash scripts/run_sft_s1k.sh 1 checkpoints/exp1

# 如果有多GPU
bash scripts/run_sft_s1k.sh 2 checkpoints/exp1
```

### 监控训练

```bash
# 终端1：监控GPU
watch -n 1 nvidia-smi

# 终端2：查看日志
tail -f checkpoints/exp1/train.log

# 终端3：启动tensorboard
tensorboard --logdir checkpoints/exp1
```

---

## 📊 第三天工作流（Day 4-5）：评估与展示

### 快速评估脚本

```bash
cat > scripts/quick_eval.py << 'EOF'
import torch
from transformers import AutoModel, AutoTokenizer

def test_model(model_path, test_prompts):
    model = AutoModel.from_pretrained(model_path, torch_dtype=torch.bfloat16, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained("Dream-org/Dream-v0-Base-7B", trust_remote_code=True)
    model = model.to("cuda").eval()
    
    for prompt in test_prompts:
        messages = [{"role": "user", "content": prompt}]
        inputs = tokenizer.apply_chat_template(messages, return_tensors="pt", return_dict=True, add_generation_prompt=True)
        input_ids = inputs.input_ids.to("cuda")
        attention_mask = inputs.attention_mask.to("cuda")
        
        output = model.diffusion_generate(
            input_ids, attention_mask=attention_mask,
            max_new_tokens=256, steps=256,
            temperature=0.2, alg="entropy"
        )
        
        response = tokenizer.decode(output.sequences[0][len(input_ids[0]):].tolist())
        print(f"\n{'='*60}")
        print(f"问题: {prompt}")
        print(f"回答: {response.split(tokenizer.eos_token)[0]}")

# 测试问题
test_prompts = [
    "3x + 7 = 22，求x",
    "写一个快速排序的Python函数",
]

print("=== Base模型 ===")
test_model("Dream-org/Dream-v0-Base-7B", test_prompts)

print("\n\n=== 微调模型 ===")
test_model("checkpoints/exp1/checkpoint-final", test_prompts)
EOF

python scripts/quick_eval.py
```

---

## 🎯 快速故障排除

### 问题：CUDA Out of Memory

```bash
# 解决方案1：减小batch size
# 修改 run_sft_s1k.sh 中的：
data.micro_batch_size_per_gpu=2  # 改为1

# 解决方案2：减小序列长度
data.max_length=2048  # 改为1024
```

### 问题：训练速度很慢

```bash
# 检查：
1. 确认使用GPU: nvidia-smi
2. 确认batch size不是太小
3. 检查数据加载: htop 看CPU
```

### 问题：Loss不下降

```bash
# 检查：
1. 学习率是否合理（2e-6）
2. 数据格式是否正确
3. 是否使用了Base模型（而非Instruct）
```

---

## 📝 每日工作日志模板

创建训练日志，记录每天的进展：

```bash
cat > docs/training_log.md << 'EOF'
# 训练日志

## Day 1 - 2025-XX-XX

### 今日目标
- [ ] 环境配置
- [ ] 数据下载和转换
- [ ] 理解Dream架构

### 完成情况
- ✅ 
- ⏳ 
- ❌ 

### 遇到的问题
1. 问题描述
   - 原因分析
   - 解决方案

### 技术笔记
- 

### 明天计划
- 

---

## Day 2 - 2025-XX-XX

...
EOF
```

---

## 💡 省时间的技巧

### 1. 使用小数据集快速验证
```python
# 只用10%数据训练1个epoch，快速验证流程
split['train'].select(range(100)).to_parquet("data/s1k_dream_format/train_small.parquet")
# 然后用train_small.parquet训练，30分钟就能看到结果
```

### 2. 使用WandB自动记录
```bash
wandb login  # 只需登录一次
# 训练日志自动上传，随时随地查看
```

### 3. 使用tmux避免断连
```bash
# 安装tmux
sudo apt install tmux

# 创建session
tmux new -s dream-train

# 运行训练
bash scripts/run_sft_s1k.sh 1 checkpoints/exp1

# 断开（Ctrl+B 然后 D）
# 重新连接：tmux attach -t dream-train
```

### 4. 准备展示模板
```bash
# 提前准备好Jupyter notebook做可视化
pip install jupyter matplotlib seaborn
jupyter notebook notebooks/
```

---

## ✅ 第一次运行检查清单

在开始完整训练前，确认：

- [ ] `test_env.py` 运行成功，GPU可用
- [ ] S1K数据下载完成，大小正确（~1000样本）
- [ ] 数据转换成功，格式正确（有prompt和response字段）
- [ ] 能用10条数据跑通训练流程（哪怕只训练几步）
- [ ] wandb/tensorboard能看到日志
- [ ] 理解了Dream的基本原理（扩散过程）

**全部打勾后，再开始完整训练！**

---

## 🎓 面试准备清单

面试前一天，准备这些：

- [ ] 训练日志（记录问题和解决方案）
- [ ] 训练曲线截图（loss下降图）
- [ ] 对比案例（base vs 微调后，3-5个例子）
- [ ] PPT或文档（15分钟讲解）
- [ ] Demo代码（现场演示）
- [ ] 架构理解笔记（能画图解释）

---

**现在就开始吧！** 从`test_env.py`开始，一步步来，相信自己！💪

如需详细步骤，查看：
- `01_overview.md` - 项目概览
- `02_detailed_steps.md` - 详细步骤
- `03_technical_notes.md` - 技术要点 