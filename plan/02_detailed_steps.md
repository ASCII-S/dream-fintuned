# Dream 7B + S1K 微调详细实施步骤

---

## 🔧 阶段1：环境准备与架构分析 (Day 1, 约4-6小时)

### 步骤 1.1：创建Python环境
```bash
# 创建虚拟环境
conda create -n dream-sft python=3.10 -y
conda activate dream-sft

# 或使用venv
python -m venv dream-env
source dream-env/bin/activate
```

### 步骤 1.2：安装依赖包
```bash
# 核心依赖
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
pip install transformers==4.46.2
pip install datasets
pip install accelerate
pip install sentencepiece
pip install protobuf

# 训练相关（Dream使用verl框架）
cd resp/Dream
pip install -e .

# 监控与可视化
pip install wandb
pip install tensorboard
pip install tqdm

# 数据处理
pip install pandas
pip install pyarrow
pip install jsonlines
```

**关键点**：
- Dream要求 `torch==2.5.1` 和 `transformers==4.46.2`，版本必须匹配
- 确保CUDA版本兼容（推荐CUDA 12.1+）

### 步骤 1.3：验证环境
```python
# test_env.py
import torch
import transformers
from transformers import AutoModel, AutoTokenizer

print(f"PyTorch版本: {torch.__version__}")
print(f"Transformers版本: {transformers.__version__}")
print(f"CUDA可用: {torch.cuda.is_available()}")
print(f"GPU数量: {torch.cuda.device_count()}")
if torch.cuda.is_available():
    print(f"GPU型号: {torch.cuda.get_device_name(0)}")
    print(f"GPU显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
```

### 步骤 1.4：理解Dream扩散模型架构

**核心概念**：
1. **扩散过程**：不同于自回归模型的逐token生成，Dream使用扩散过程同时生成多个token
2. **Remasking策略**：控制token生成顺序的策略
   - `origin`: 随机顺序
   - `entropy`: 基于熵的置信度
   - `maskgit_plus`: top1置信度
   - `topk_margin`: top1-top2边际置信度

**关键文件阅读**：
```bash
# 阅读这些文件了解架构
- resp/Dream/src/diffllm/modeling_diffllm.py  # 模型核心代码
- resp/Dream/src/diffllm/configuration_diffllm.py  # 配置文件
- resp/Dream/src/trainer/fsdp_sft_trainer.py  # 训练器
```

**动手实验**：
```python
# test_inference.py - 测试Dream推理
import torch
from transformers import AutoModel, AutoTokenizer

model_path = "Dream-org/Dream-v0-Instruct-7B"
model = AutoModel.from_pretrained(model_path, torch_dtype=torch.bfloat16, trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = model.to("cuda").eval()

messages = [{"role": "user", "content": "解释什么是扩散模型"}]
inputs = tokenizer.apply_chat_template(messages, return_tensors="pt", return_dict=True, add_generation_prompt=True)
input_ids = inputs.input_ids.to("cuda")
attention_mask = inputs.attention_mask.to("cuda")

output = model.diffusion_generate(
    input_ids,
    attention_mask=attention_mask,
    max_new_tokens=256,
    steps=128,
    temperature=0.2,
    alg="entropy"
)

print(tokenizer.decode(output.sequences[0][len(input_ids[0]):].tolist()).split(tokenizer.eos_token)[0])
```

### 步骤 1.5：分析S1K数据集

```python
# explore_s1k.py - 探索S1K数据集
from datasets import load_dataset

# 加载数据集
dataset = load_dataset("simplescaling/s1K-1.1", split="train")

print(f"数据集大小: {len(dataset)}")
print(f"数据字段: {dataset.column_names}")
print("\n第一个样本:")
print(dataset[0])

# 统计分析
import pandas as pd
df = dataset.to_pandas()

# 分析数据来源
print("\n数据来源分布:")
print(df['source'].value_counts())

# 分析文本长度
df['question_length'] = df['question'].apply(len)
df['answer_length'] = df['answer'].apply(len)
print(f"\n问题平均长度: {df['question_length'].mean():.0f} 字符")
print(f"答案平均长度: {df['answer_length'].mean():.0f} 字符")
```

**预期数据格式**：
```json
{
  "question": "问题文本",
  "answer": "带推理过程的答案",
  "source": "数据来源（如AIME, GPQA等）",
  "thinking": "推理轨迹（可选）"
}
```

---

## 📊 阶段2：数据准备 (Day 2, 约4-6小时)

### 步骤 2.1：下载S1K-1.1数据集

```python
# download_s1k.py
from datasets import load_dataset
import os

# 下载数据集
dataset = load_dataset("simplescaling/s1K-1.1")
print(f"数据集splits: {dataset.keys()}")

# 保存到本地
save_dir = "data/s1k"
os.makedirs(save_dir, exist_ok=True)
dataset.save_to_disk(save_dir)
print(f"数据集已保存到: {save_dir}")
```

### 步骤 2.2：数据格式转换

Dream的SFT训练需要以下格式：
- `prompt`: list of messages（除了最后一条assistant消息）
- `response`: str（assistant的回复内容）

```python
# prepare_data_for_dream.py
from datasets import load_dataset, Dataset
import json
import os

def convert_s1k_to_dream_format(sample):
    """
    将S1K格式转换为Dream训练格式
    
    S1K格式:
    {
        "question": "...",
        "answer": "...",  # 可能包含<think>...</think>标签
    }
    
    Dream格式:
    {
        "prompt": [{"role": "user", "content": "..."}],
        "response": "..."
    }
    """
    question = sample.get('question', sample.get('prompt', ''))
    answer = sample.get('answer', sample.get('response', ''))
    
    # 构造prompt（chat格式）
    prompt = [
        {
            "role": "system",
            "content": "You are a helpful AI assistant. Think step by step and provide detailed reasoning."
        },
        {
            "role": "user", 
            "content": question
        }
    ]
    
    # response就是答案（包含推理过程）
    response = answer
    
    return {
        "prompt": prompt,
        "response": response
    }

# 加载S1K数据集
dataset = load_dataset("simplescaling/s1K-1.1", split="train")
print(f"原始数据量: {len(dataset)}")

# 转换格式
converted_data = [convert_s1k_to_dream_format(sample) for sample in dataset]
converted_dataset = Dataset.from_list(converted_data)

# 划分训练集和验证集（90/10分割）
split_dataset = converted_dataset.train_test_split(test_size=0.1, seed=42)
train_dataset = split_dataset['train']
val_dataset = split_dataset['test']

print(f"训练集大小: {len(train_dataset)}")
print(f"验证集大小: {len(val_dataset)}")

# 保存为parquet格式（Dream训练器支持）
output_dir = "data/s1k_dream_format"
os.makedirs(output_dir, exist_ok=True)
train_dataset.to_parquet(f"{output_dir}/train.parquet")
val_dataset.to_parquet(f"{output_dir}/val.parquet")

print(f"数据已保存到: {output_dir}")
print("\n示例数据:")
print(json.dumps(train_dataset[0], indent=2, ensure_ascii=False))
```

### 步骤 2.3：数据质量检查

```python
# validate_data.py
import pandas as pd
from transformers import AutoTokenizer

# 加载数据
train_df = pd.read_parquet("data/s1k_dream_format/train.parquet")
val_df = pd.read_parquet("data/s1k_dream_format/val.parquet")

# 初始化tokenizer
tokenizer = AutoTokenizer.from_pretrained("Dream-org/Dream-v0-Base-7B", trust_remote_code=True)

def check_length(row):
    """检查token长度"""
    prompt = row['prompt']
    response = row['response']
    messages = prompt + [{"role": "assistant", "content": response}]
    tokens = tokenizer.apply_chat_template(messages, tokenize=True)
    return len(tokens)

# 计算token长度
train_df['token_length'] = train_df.apply(check_length, axis=1)

print("Token长度统计:")
print(train_df['token_length'].describe())
print(f"\n超过2048 tokens的样本数: {(train_df['token_length'] > 2048).sum()}")
print(f"超过1024 tokens的样本数: {(train_df['token_length'] > 1024).sum()}")

# 过滤过长的样本（可选）
max_length = 2048
filtered_train_df = train_df[train_df['token_length'] <= max_length]
print(f"\n过滤后训练集大小: {len(filtered_train_df)} (原始: {len(train_df)})")

# 保存过滤后的数据
if len(filtered_train_df) < len(train_df):
    filtered_train_df.drop(columns=['token_length']).to_parquet("data/s1k_dream_format/train_filtered.parquet")
```

---

## 🚀 阶段3：微调训练 (Day 3-4, 约8-12小时)

### 步骤 3.1：配置训练脚本

创建训练配置脚本：

```bash
# scripts/run_sft_s1k.sh
#!/bin/bash

set -x

if [ "$#" -lt 2 ]; then
    echo "Usage: scripts/run_sft_s1k.sh <nproc_per_node> <save_path>"
    exit 1
fi

nproc_per_node=$1
save_path=$2

# 数据路径
DATA_DIR="data/s1k_dream_format"
TRAIN_FILE="$DATA_DIR/train.parquet"
VAL_FILE="$DATA_DIR/val.parquet"

# 模型路径
MODEL_PATH="Dream-org/Dream-v0-Base-7B"

# 训练参数
LR=2e-6
BATCH_SIZE=4  # 根据GPU显存调整
MAX_LENGTH=2048
EPOCHS=3

# 启动训练
torchrun --standalone --nnodes=1 --nproc_per_node=$nproc_per_node \
    -m src.trainer.fsdp_sft_trainer \
    diffusion.time_reweighting=cart \
    data.train_files=$TRAIN_FILE \
    data.val_files=$VAL_FILE \
    data.max_length=$MAX_LENGTH \
    data.prompt_key=prompt \
    data.response_key=response \
    data.truncation=right \
    optim.lr=$LR \
    data.micro_batch_size_per_gpu=$BATCH_SIZE \
    data.enable_perbatch_cutoff=True \
    data.perbatch_cutoff_type=random_with_input_pad \
    +data.perbatch_cutoff=True \
    model.partial_pretrain=$MODEL_PATH \
    model.trust_remote_code=True \
    model.enable_gradient_checkpointing=True \
    trainer.default_local_dir=$save_path \
    trainer.project_name=dream-s1k-sft \
    trainer.experiment_name=s1k-$(date +%Y%m%d-%H%M%S) \
    trainer.logger=['console','wandb'] \
    trainer.total_epochs=$EPOCHS \
    trainer.save_freq=500
```

### 步骤 3.2：启动训练

```bash
# 单GPU训练
cd resp/Dream
bash ../../scripts/run_sft_s1k.sh 1 ../../checkpoints/s1k-sft

# 多GPU训练（如果有8块GPU）
bash ../../scripts/run_sft_s1k.sh 8 ../../checkpoints/s1k-sft
```

**根据GPU显存调整参数**：

| GPU配置 | batch_size | gradient_checkpointing | 预计训练时间 |
|---------|------------|------------------------|-------------|
| 1x 4090 (24GB) | 2 | True | ~8-10小时 |
| 2x 4090 (48GB) | 4 | True | ~4-5小时 |
| 1x A100 (40GB) | 4 | True | ~5-6小时 |
| 8x GPU | 8 | False | ~1-2小时 |

### 步骤 3.3：监控训练

**使用WandB监控**：
```bash
# 登录wandb
wandb login

# 查看训练曲线
# 访问 https://wandb.ai/your-username/dream-s1k-sft
```

**使用TensorBoard监控**：
```bash
# 启动tensorboard
tensorboard --logdir checkpoints/s1k-sft --port 6006

# 在浏览器访问 http://localhost:6006
```

**命令行监控**：
```bash
# 监控GPU使用
watch -n 1 nvidia-smi

# 查看训练日志
tail -f checkpoints/s1k-sft/train.log
```

### 步骤 3.4：处理常见问题

**问题1：OOM (Out of Memory)**
```bash
# 解决方案：
# 1. 减小batch_size（从4改到2或1）
# 2. 启用gradient_checkpointing
# 3. 减小max_length（从2048改到1024）
# 4. 使用更少的GPU进程
```

**问题2：训练速度慢**
```bash
# 解决方案：
# 1. 确认使用了bfloat16
# 2. 检查数据加载是否是瓶颈
# 3. 增加batch_size（如果显存允许）
```

---

## 📈 阶段4：评估与分析 (Day 4-5, 约4-6小时)

### 步骤 4.1：加载微调后的模型

```python
# load_finetuned_model.py
import torch
from transformers import AutoModel, AutoTokenizer

# 加载微调后的checkpoint
checkpoint_path = "checkpoints/s1k-sft/checkpoint-final"  # 根据实际路径调整
model = AutoModel.from_pretrained(checkpoint_path, torch_dtype=torch.bfloat16, trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained("Dream-org/Dream-v0-Base-7B", trust_remote_code=True)

model = model.to("cuda").eval()
print("模型加载成功!")
```

### 步骤 4.2：定性评估

```python
# qualitative_eval.py - 对比测试
import torch
from transformers import AutoModel, AutoTokenizer

def generate_response(model, tokenizer, prompt, max_tokens=512):
    messages = [{"role": "user", "content": prompt}]
    inputs = tokenizer.apply_chat_template(messages, return_tensors="pt", return_dict=True, add_generation_prompt=True)
    input_ids = inputs.input_ids.to("cuda")
    attention_mask = inputs.attention_mask.to("cuda")
    
    output = model.diffusion_generate(
        input_ids,
        attention_mask=attention_mask,
        max_new_tokens=max_tokens,
        steps=256,
        temperature=0.2,
        alg="entropy"
    )
    
    response = tokenizer.decode(output.sequences[0][len(input_ids[0]):].tolist())
    return response.split(tokenizer.eos_token)[0]

# 加载base模型和微调模型
base_model = AutoModel.from_pretrained("Dream-org/Dream-v0-Base-7B", torch_dtype=torch.bfloat16, trust_remote_code=True).to("cuda").eval()
finetuned_model = AutoModel.from_pretrained("checkpoints/s1k-sft/checkpoint-final", torch_dtype=torch.bfloat16, trust_remote_code=True).to("cuda").eval()
tokenizer = AutoTokenizer.from_pretrained("Dream-org/Dream-v0-Base-7B", trust_remote_code=True)

# 测试问题
test_prompts = [
    "一个数的3倍加上5等于26，这个数是多少？",
    "解释为什么2+2=4",
    "写一个Python函数来计算斐波那契数列的第n项"
]

for prompt in test_prompts:
    print(f"\n{'='*60}")
    print(f"问题: {prompt}")
    print(f"\n[Base模型回答]:")
    print(generate_response(base_model, tokenizer, prompt))
    print(f"\n[微调后模型回答]:")
    print(generate_response(finetuned_model, tokenizer, prompt))
```

### 步骤 4.3：定量评估

```python
# quantitative_eval.py - 在验证集上评估
from datasets import load_dataset
from tqdm import tqdm
import json

# 加载验证集
val_dataset = load_dataset("parquet", data_files="data/s1k_dream_format/val.parquet", split="train")

results = []
correct = 0

for sample in tqdm(val_dataset):
    question = sample['prompt'][-1]['content']
    ground_truth = sample['response']
    
    # 生成答案
    prediction = generate_response(finetuned_model, tokenizer, question)
    
    # 简单的答案匹配（实际需要更复杂的评估逻辑）
    is_correct = evaluate_answer(prediction, ground_truth)  # 需要实现
    if is_correct:
        correct += 1
    
    results.append({
        "question": question,
        "ground_truth": ground_truth,
        "prediction": prediction,
        "correct": is_correct
    })

accuracy = correct / len(val_dataset)
print(f"\n验证集准确率: {accuracy:.2%}")

# 保存结果
with open("eval_results.json", "w") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)
```

### 步骤 4.4：准备展示材料

创建展示文档：
```markdown
# Dream 7B + S1K 微调项目成果报告

## 1. 项目概述
- 目标：使用S1K数据集微调Dream 7B模型
- 数据：1000个高质量推理样本
- 训练：3 epochs，学习率2e-6

## 2. 训练过程
- GPU配置：[填写]
- 训练时长：[填写]
- 最终Loss：[填写]
- 训练曲线：[插入图片]

## 3. 评估结果
### 定量结果
- 验证集准确率：[填写]
- 对比base模型提升：[填写]

### 定性案例
[展示3-5个对比案例]

## 4. 技术挑战与解决
1. 挑战1：[描述]
   - 解决方案：[描述]
2. ...

## 5. 架构理解
- 扩散模型vs自回归模型的差异
- Remasking策略的影响
- ...

## 6. 未来改进方向
- 超参数优化
- 数据增强
- ...
```

---

## ✅ 检查清单

完成以下所有任务后，您就可以参加终面了：

- [ ] **环境配置**：所有依赖安装成功
- [ ] **架构理解**：能解释Dream扩散模型的工作原理
- [ ] **数据准备**：S1K数据成功转换为Dream格式
- [ ] **训练完成**：至少训练1-3个epoch并保存checkpoint
- [ ] **评估完成**：对比base模型和微调模型的性能
- [ ] **文档完整**：记录所有问题、解决方案和思考过程
- [ ] **Demo准备**：能现场展示微调前后的效果

---

## 📞 遇到问题怎么办？

1. **查看官方文档**：Dream和S1 的GitHub README
2. **查看Issues**：GitHub仓库的Issues页面
3. **使用AI助手**：Claude、GPT、Gemini 2.5 Pro等
4. **联系面试官**：微信/电话 17274608033

**记住**：过程比结果更重要！记录您的思考过程、遇到的问题和解决方案。

---

下一步：查看 `03_technical_notes.md` 了解关键技术细节。 