# Dream 7B + S1K 微调项目概览

## 📋 项目目标

使用 **Dream 7B** 扩散型大语言模型 (dLLM)，通过 **S1K** 数据集进行监督微调 (SFT)，提升模型在推理任务上的表现。

---

## 🔑 核心要素

### 1. 基础模型：Dream 7B
- **类型**：扩散型大语言模型 (Diffusion LLM)
- **参数量**：7B
- **模型版本**：
  - `Dream-org/Dream-v0-Base-7B` (基础版本，用于微调)
  - `Dream-org/Dream-v0-Instruct-7B` (指令版本，参考对比)
- **特点**：
  - 使用扩散过程生成文本，而非传统的自回归方式
  - 支持自定义生成策略（remasking strategies）
  - 上下文长度：2048 tokens
  - 最小GPU需求：20GB显存

### 2. 微调数据集：S1K
- **来源**：Simple test-time scaling 项目
- **版本**：s1K-1.1 (推荐，使用r1生成的推理轨迹)
- **数据量**：1000个高质量示例
- **数据特点**：
  - 包含复杂推理任务
  - 每个样本包含详细的推理过程（thinking traces）
  - 涵盖数学、编程、逻辑推理等领域
- **HuggingFace链接**：`simplescaling/s1K-1.1`

---

## 🎯 项目里程碑

### 阶段1：环境准备与分析 (Day 1)
- [ ] 环境配置与依赖安装
- [ ] 理解Dream扩散模型架构
- [ ] 分析S1K数据集格式
- [ ] 测试模型基础推理能力

### 阶段2：数据准备 (Day 2)
- [ ] 下载并处理S1K-1.1数据集
- [ ] 数据格式转换（转为Dream所需格式）
- [ ] 数据验证与统计分析
- [ ] 划分训练集和验证集

### 阶段3：微调实验 (Day 3-4)
- [ ] 配置训练超参数
- [ ] 启动SFT训练
- [ ] 监控训练过程（loss、GPU使用率等）
- [ ] 保存checkpoint

### 阶段4：评估与优化 (Day 4-5)
- [ ] 评估微调后模型性能
- [ ] 对比base模型表现
- [ ] 分析改进效果
- [ ] 准备展示材料

---

## 💰 资源预算

- **计算预算**：100元人民币（可报销）
- **推荐平台**：
  - AutoDL（性价比高）
  - 阿里云PAI
  - 腾讯云GPU
- **GPU需求**：
  - 最低：1x RTX 4090 (24GB)
  - 推荐：2x RTX 4090 或 1x A100 (40GB)
  - 理想：8x GPU (如果预算允许)

---

## 📊 预期成果

1. **微调后的模型checkpoint**
2. **训练日志与曲线**（loss, learning rate等）
3. **评估结果对比报告**
4. **技术文档**：
   - 架构理解笔记
   - 遇到的问题与解决方案
   - 超参数选择理由
5. **演示Demo**：展示微调前后的效果对比

---

## 🚨 风险与挑战

### 技术挑战
1. **扩散模型理解**：与传统Transformer不同，需要理解扩散过程
2. **显存优化**：7B模型 + 长序列可能导致OOM
3. **训练时间**：计算预算有限，需要高效利用资源
4. **数据格式适配**：S1K格式需要转换为Dream训练格式

### 解决策略
1. 使用梯度检查点 (gradient checkpointing)
2. 混合精度训练 (bfloat16)
3. 合理设置batch size和序列长度
4. 参考官方示例代码进行适配

---

## 📚 核心参考资料

### 官方文档
- [Dream GitHub](https://github.com/HKUNLP/Dream)
- [Dream Blog](https://hkunlp.github.io/blog/2025/dream/)
- [Dream Paper](https://arxiv.org/abs/2508.15487)
- [S1 GitHub](https://github.com/simplescaling/s1)
- [S1 Paper](https://arxiv.org/abs/2501.19393)

### 关键代码位置
- 训练脚本：`resp/Dream/examples/run_sft_tulu3.sh`
- 数据准备：`resp/Dream/examples/prepare_tulu3.py`
- 模型代码：`resp/Dream/src/diffllm/`
- S1数据处理：`resp/s1/data/`

---

## 🔄 下一步行动

查看 `02_detailed_steps.md` 获取详细的实施步骤。 