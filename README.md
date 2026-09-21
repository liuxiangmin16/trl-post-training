# Qwen3-0.6B 后训练实验（SFT → GRPO）

SFT  → 教模型输出 <answer></answer> 格式。

GRPO → 提升模型解题能力。

在 GSM8K 上后训练之后，对比 **原模型 / SFT / GRPO** 的解题能力。结论先说：**基座已经会做题；SFT 把格式教对了，但把长推理压短，正确率下降；本轮 GRPO 没有把正确率拉回来。**

## 评测结果

数据：GSM8K **test** 前 20 题（SFT 用 train[:500]，GRPO 用 train[500:1000]，与 test 无重叠）。

三套权重同一条 prompt（要求 `<answer>`）、贪心解码、`max_new_tokens=2048`。预测优先取 `<answer>`，没有再取 `\boxed{}`。明细见 `data/eval/`。


| 模型                     | 正确      | 准确率     | 使用 `<answer>` | 使用 `\boxed{}` | 平均生成 token |
| ---------------------- | ------- | ------- | ------------- | ------------- | ---------- |
| 原模型 Qwen3-0.6B         | 10 / 20 | **50%** | 10%           | 50%           | **1446**   |
| SFT LoRA               | 5 / 20  | 25%     | 85%           | 0%            | 389        |
| SFT merged + GRPO LoRA | 5 / 20  | 25%     | 90%           | 0%            | 298        |


样本量只有 20，数字会抖，但方向一致：

- 原模型用长思维链 + `\boxed{}`，这 20 题对一半。
- SFT 后几乎都收口成 `<answer>`，平均长度从 ~1400 token 掉到 ~390，准确率减半。
- GRPO 后格式更稳一点，生成更短，**正确题数与 SFT 相同**。

这与训练日志一致：GRPO 全程 `format_reward ≈ 0.5`，`correctness_reward` 均值约 0.33，曲线走平，没有相对 SFT 的明显抬升。

## 怎么理解

GSM8K 金标解很短（中位大约 3 行 / 40 词）。SFT 是在模仿这种短解写进 `<think>`，再加 `<answer>gold</answer>`。Qwen3 原来会把已知条件写清楚再算；短解会跳步，简单题也会算错（例如把「红是蓝的两倍」写成「总数对半」）。

GRPO 只按最终数字打分，不奖励推理长度或过程，所以不会把长 CoT 找回来。

## 优化方向

1. **把推理长度加入奖励。** 现在 GRPO 只有 `format_reward`（有没有 `<answer>`）和 `correctness_reward`（数字对不对）。短而错、短而对都会给予奖励。可以加一项长度奖励（例如 completion token 数落在基座量级、或相对组内均值），让模型不要把 `<think>` 压成两三行。要注意设上限，避免为刷分而注水。
2. **换更详细的推理过程数据，尝试 OpenR1-Math。** SFT 目前模仿 GSM8K 金标短解。OpenR1-Math（如 [open-r1/OpenR1-Math-220k](https://huggingface.co/datasets/open-r1/OpenR1-Math-220k)）带长 CoT，更接近 Qwen3 原来的写法。用它做 SFT（或只 SFT 长推理、格式用少量标签样本），再 GRPO，比在短解上继续 RL 更对症。



## 训练设置


| 阶段    | 数据                                                                     | 主要超参                           | 权重                                          |
| ----- | ---------------------------------------------------------------------- | ------------------------------ | ------------------------------------------- |
| SFT   | GSM8K train[:500]，assistant = `<think>短解</think><answer>gold</answer>` | 100 step，lr 2e-4，LoRA r=16     | `outputs/sft/qwen3-0.6b-lora-math-format`   |
| merge | 把 SFT LoRA 合进基座                                                        | `scripts/merge_lora.py`        | `outputs/sft/qwen3-0.6b-merged-math-format` |
| GRPO  | GSM8K train[500:1000]                                                  | 500 step，G=8，lr 2e-6，2×A40 DDP | `outputs/grpo/qwen3-0.6b-gsm8k-from-sft`    |


GRPO 必须叠在 **merged SFT** 上，不能直接挂在原版 Qwen 上。

## 复现评测

```bash
CUDA_VISIBLE_DEVICES=2 python scripts/eval_math.py \
  --dataset data/raw/openai/gsm8k \
  --split test \
  --limit 20 \
  --base_model /rainbow/liuxm/learn/qwen3-0.6B \
  --sft_adapter outputs/sft/qwen3-0.6b-lora-math-format \
  --grpo_base outputs/sft/qwen3-0.6b-merged-math-format \
  --grpo_adapter outputs/grpo/qwen3-0.6b-gsm8k-from-sft \
  --output_dir data/eval
```

单题对比用 `scripts/infer.py`。默认贪心解码；加 `--temperature 0.7` 才会换不同输出。

- 原模型：`--base_model .../qwen3-0.6B`
- SFT：上面再加 `--adapter outputs/sft/qwen3-0.6b-lora-math-format`
- GRPO：`--base_model` 用 merged SFT，`--adapter` 用 GRPO LoRA

全量 test（1319 题）把 `--limit` 改成 `-1`。

## 仓库

```
configs/          SFT / GRPO yaml，DeepSpeed json
scripts/          数据、训练、merge、推理、评测
data/raw/         GSM8K parquet
data/processed/   SFT / DPO / GRPO jsonl
data/eval/        summary.json 与各模型 preds.jsonl
```

