# Qwen3-0.6B 微调 · 安全 + 数学 + 通用多目标优化

以 **Qwen3-0.6B** 为基座微调，在保持通用能力的前提下**同步提升安全对齐与数学推理**。

- **打分**：`总分 = 安全 × 40% + 数学 × 30% + 通用 × 30%`（三个测试集官方不公开）
- **约束**：基座固定为 [Qwen/Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B)，参数量 0.6B 不可变
- **产出**：一个微调后的 0.6B 模型 + 一页报告

---

## 🎯 最终成果

**模型已发布在 HuggingFace**：[**fengkai-zhu/qwen3-0.6b-lora**](https://huggingface.co/fengkai-zhu/qwen3-0.6b-lora)（子目录 `final_model_v10-4b/`）

| 维度 | 权重 | 基座 Qwen3-0.6B | **本模型(v10-4b)** | 提升 |
|------|------|------|------|------|
| 🔴 安全（有害拒答率）| 40% | 15.1% | **82.2%** | +67 |
| 🟢 数学（GSM8K）| 30% | 64.2% | **65.7%** | +1.5 |
| 🔵 通用（MMLU）| 30% | 35.0% | **46.6%** | +11.6 |
| **加权代理分** | | 0.358 | **0.666** | — |

> 评测口径为自建代理评测集（0-shot），详见 [eval/EXPERIMENTS.md](eval/EXPERIMENTS.md)。

### 如何使用

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

repo = "fengkai-zhu/qwen3-0.6b-lora"
sub = "final_model_v10-4b"          # 权重在此子目录下
model = AutoModelForCausalLM.from_pretrained(repo, subfolder=sub, dtype=torch.bfloat16)
tok = AutoTokenizer.from_pretrained(repo, subfolder=sub)

msgs = [{"role": "user", "content": "A robe takes 2 bolts of blue fiber and half that much white fiber. How many bolts in total?"}]
text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
out = model.generate(**tok(text, return_tensors="pt"), max_new_tokens=256, do_sample=False)
print(tok.decode(out[0], skip_special_tokens=True))
```

> **注意**：① 权重在 `final_model_v10-4b/` 子目录，加载要带 `subfolder`。② 我们用**非 thinking** 数据训练，推理建议 `enable_thinking=False`（thinking 模式也不塌，数学 62.2%）。

---

## 🔬 这个模型是怎么来的（溯源）

从基座到最终模型，核心是 **3 步流水线**（完整 12 版探索见 [eval/EXPERIMENTS.md](eval/EXPERIMENTS.md)）：

```
Qwen3-0.6B(基座)
   │  ① 混合 SFT(LoRA)：安全(Aegis拒答) + 数学 + 通用(Tulu3) 一次性混训
   │     配比 50:30:20，非thinking，防多任务干扰
   ▼
   │  ② 数学用"大模型蒸馏"数据：Qwen3-4B 解 GSM8K → 留答对的高质量 CoT
   │     突破 0.6B 自蒸馏上限(61.9 → 65.7)；验证了"容量鸿沟"(4B比8B更适合0.6B学生)
   ▼
最终模型 v10-4b（LoRA adapter 合并进基座 → 完整模型）
```

**关键决策依据**（每一步都有实证/调研支撑）：
1. **安全优先**：基线才 15%、权重最高(40%) → 用 Aegis refusals 训，是最大涨分杠杆。
2. **非 thinking / 短 CoT**：0.6B 长思考有"山谷效应"，直接分步答更准。
3. **自蒸馏修数据质量**：原始 GSM8K 解答过简短会把数学"教差"，改用模型自己/更强老师的正确解答。
4. **大模型蒸馏破上限**：小模型能模仿比自己强的老师，但老师别太大（4B 对 0.6B 是甜点，8B 反而差）。

**探索过但排除的方向**（详见实验文档）：
- ❌ **修过度拒答**：安全训到 85% 的代价是过度拒答 87%，三种方法(FalseReject/削配比/自蒸馏)都治不了 → 0.6B 硬伤，但标准安全评测通常不计此项。
- ❌ **模型合并(TIES)**：3 专家合并冲突严重(数学崩到 49)，输给混合 SFT。
- ❌ **GRPO 强化学习**：稳但无增益，因数学已在天花板 → 验证 65% 是真上限。

**最终结论**：SFT / 蒸馏 / 合并 / RL 四种方法全部收敛到加权 ~0.66-0.67，说明已触及这个 0.6B 在本任务上的**能力上限**。

---

## 📁 仓库导航

**文档（按推荐阅读顺序）：**
| 文件 | 内容 |
|------|------|
| [新手入门指南.md](新手入门指南.md) | 🌱 **零基础入门**：LLM/微调/SFT/三种能力/坑/流程（含比喻+技术细节）|
| [第一部分实施方案.md](第一部分实施方案.md) | 📋 **技术方案**：6 步流程、数据选型、训练策略、防坑纪律 |
| [eval/EXPERIMENTS.md](eval/EXPERIMENTS.md) | 🔬 **完整实验详录**：12 版 + 3 方法路线、数据/配比/动机/结论（最重要）|
| [eval/RESULTS.md](eval/RESULTS.md) | 📊 结果表 · [eval/BASELINE.md](eval/BASELINE.md) 基线 · [eval/RESEARCH_NOTES.md](eval/RESEARCH_NOTES.md) 调研 |
| [夏令营考核.md](夏令营考核.md) | 原始考核要求 |

**代码：**
| 目录/文件 | 用途 |
|------|------|
| [scripts/prepare_data.py](scripts/prepare_data.py) | 数据转 LLaMA-Factory 格式 + 混合配比 |
| [scripts/rejection_sample_math.py](scripts/rejection_sample_math.py) | 拒绝采样造数学数据（自蒸馏 / 大模型蒸馏通用）|
| [scripts/ties_merge.py](scripts/ties_merge.py) | 手写 TIES 模型合并（mergekit 不支持 Qwen3）|
| [scripts/grpo_math.py](scripts/grpo_math.py) | 数学 GRPO 强化学习（RLVR）|
| [configs/](configs/) | 各版本 LLaMA-Factory 训练配置（v1/v2/v3 等）|
| [eval/](eval/) | 四方向评测：数学/安全/通用/过度拒答 + 5-shot MMLU |

---

## 🧪 复现

**评测**（模型权重从 HF 拉取，数据集需另备）：
```bash
cd eval
python eval_math.py    --model <模型路径> --data gsm8k_test.parquet --out math.json
python eval_safety.py  --model <模型路径> --data safety_eval.json    --out safety.json
python eval_general.py --model <模型路径> --data mmlu_test.parquet   --out general.json
```

**训练流程**（需 GPU + LLaMA-Factory）：
1. `prepare_data.py` 造混合数据 → 2. `llamafactory-cli train configs/xxx.yaml` → 3. `llamafactory-cli export` 合并 LoRA → 4. `eval/` 评测。
   数学数据可先用 `rejection_sample_math.py` 配 Qwen3-4B 老师蒸馏。

> 数据集、模型权重、`CLAUDE.md`（含内网信息）均不入库，见 `.gitignore`。

---

## 环境

- **训练**：8×80GB GPU，LLaMA-Factory 0.9.3 + torch 2.3.1 + transformers 4.51.3
- **微调方式**：LoRA (rank 16)；数据"小而精"（每任务几千条），非 thinking，lr 1e-4
