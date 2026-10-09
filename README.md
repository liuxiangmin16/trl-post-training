# Qwen3-0.6B 后训练实验（SFT → GRPO）

SFT  → 教模型输出  格式。

GRPO → 提升模型解题能力。

在 GSM8K 上后训练之后，对比 **原模型 / SFT / GRPO** 的解题能力。结论先说：**基座已经会做题；SFT 把格式教对了，但把长推理压短，正确率下降；本轮 GRPO 没有把正确率拉回来。**

## 评测结果

数据：GSM8K **test** 前 20 题（SFT 用 train[:500]，GRPO 用 train[500:1000]，与 test 无重叠）。

三套权重同一条 prompt（要求 `<answer>`）、贪心解码、`max_new_tokens=2048`。预测优先取 `<answer>`，没有再取 `\boxed{}`。回答完成率是生成 token 数小于上限、没有被截断的比例。明细见 `data/eval/gsm8k-finetuned-model/`。


| 模型                     | 正确      | 准确率     | 使用 `<answer>` | 使用 `\boxed{}` | 平均生成 token | 回答完成率        |
| ---------------------- | ------- | ------- | ------------- | ------------- | ---------- | ------------ |
| 原模型 Qwen3-0.6B         | 10 / 20 | **50%** | 10%           | 50%           | **1446**   | 11 / 20（55%） |
| SFT LoRA               | 5 / 20  | 25%     | 85%           | 0%            | 389        | 17 / 20（85%） |
| SFT merged + GRPO LoRA | 5 / 20  | 25%     | 90%           | 0%            | 298        | 18 / 20（90%） |


样本量只有 20，数字会抖，但方向一致：

- 原模型用长思维链 + `\boxed{}`，这 20 题对一半。
- SFT 后几乎都收口成 `<answer>`，平均长度从 ~1400 token 掉到 ~390，准确率减半。
- GRPO 后格式更稳一点，生成更短，**正确题数与 SFT 相同**。

这与训练日志一致：GRPO 全程 `format_reward ≈ 0.5`，`correctness_reward` 均值约 0.33，曲线走平，没有相对 SFT 的明显抬升。

## 怎么理解

GSM8K 金标解很短（中位大约 3 行 / 40 词）。SFT 是在模仿这种短解写进 `<think>`，再加 `<answer>gold</answer>`。Qwen3 原来会把已知条件写清楚再算；短解会跳步，简单题也会算错（例如把「红是蓝的两倍」写成「总数对半」）。

GRPO 只按最终数字打分，不奖励推理长度或过程，所以不会把长 CoT 找回来。

## 优化方向

1. **把推理长度加入奖励。** 现在 GRPO 只有 `format_reward`（有没有 `<answer>`）和 `correctness_reward`（数字对不对）。短而错、短而对都会给予奖励。可以加一项长度奖励（例如 completion token 数落在基座量级、或相对组内均值），让模型不要把 `<think>` 压成两三行。要注意设上限，避免为刷分而注水。(实验结果表明，这个优化方向没有效果，长度加入奖励不足以扭转数据集的短板)
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
  --output_dir data/eval/gsm8k-finetuned-model
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



## 优化方向 2 的结果

用 OpenR1-Math 的长 CoT 做 SFT 和 GRPO做后训练。评测仍是 GSM8K test 前 20 题，同一条 prompt、贪心解码，`max_new_tokens`参数由默认的2048增加到4096。明细见 `data/eval/OpenR1-finetuned-model/`。

结论：**长推理留下来了，准确率回到基座并多对 1 题；GRPO 没有再抬上去。不好的地方是回答用指定格式的比例大幅下降，指令遵循效果变差。**


| 实验            | 模型   | 正确      | 准确率     | 使用 `<answer>` | 使用 `\boxed{}` | 平均生成 token | 回答完成率        |
| ------------- | ---- | ------- | ------- | ------------- | ------------- | ---------- | ------------ |
| GSM8K · 2048  | 原模型  | 10 / 20 | 50%     | 10%           | 50%           | 1446       | 11 / 20（55%） |
| GSM8K · 2048  | SFT  | 5 / 20  | 25%     | 85%           | 0%            | 389        | 17 / 20（85%） |
| GSM8K · 2048  | GRPO | 5 / 20  | 25%     | 90%           | 0%            | 298        | 18 / 20（90%） |
| OpenR1 · 4096 | 原模型  | 10 / 20 | 50%     | 10%           | 55%           | 2264       | 13 / 20（65%） |
| OpenR1 · 4096 | SFT  | 11 / 20 | **55%** | 45%           | 30%           | **1838**   | 15 / 20（75%） |
| OpenR1 · 4096 | GRPO | 10 / 20 | 50%     | 45%           | 30%           | 1989       | 15 / 20（75%） |


- OpenR1 SFT 正确 11 / 20，高于 GSM8K SFT 的 5 / 20，也高于同一次评测里的基座（10 / 20）。平均长度留在 1838 token，没有再掉到三四百。
- 回答完成率：OpenR1 SFT、GRPO 都是 15 / 20（75%），基座是 13 / 20（65%）。GSM8K 短解完成率 85%–90%，是因为生成只有约 300 token，很少撞上 2048。顶满上限的题在这 20 题里全部算错；OpenR1 SFT、GRPO 各有 5 题、基座有 7 题没写完。
- 格式没有短解那轮稳。`<answer>` 约占 45%，另有约 30% 的预测来自 `\boxed{}`。
- GRPO 正确 10 / 20，与基座相同，比 SFT 少 1 题。完成率也停在 75%。

```bash
CUDA_VISIBLE_DEVICES=2 python scripts/eval_math.py \
  --dataset data/raw/openai/gsm8k \
  --split test \
  --limit 20 \
  --max_new_tokens 4096 \
  --base_model /rainbow/liuxm/learn/qwen3-0.6B \
  --sft_adapter outputs/sft/qwen3-0.6b-lora-openr1-format \
  --grpo_base outputs/sft/qwen3-0.6b-merged-openr1-format \
  --grpo_adapter outputs/grpo/qwen3-0.6b-openr1-from-sft \
  --output_dir data/eval/OpenR1-finetuned-model
```



## 问题分析



### 指令遵循效果变差，回答用指定格式 `<answer>` 的比例大幅下降。

  原因是 SFT 训练时 `max_length` 没有从 512 调整为 4096，而标签在长文本末尾，训练时被截掉了，模型实际看到的收口是 `\boxed{}`。

  继续优化：

1. 修复 SFT 训练参数，`max_length` 调整为 4096。
2. 训练数据预处理时，思维链里的 `\boxed{...}` 拆成普通文本，最终答案只留在 `<answer>`。



### GRPO 正确率没涨。

  从日志看，是因为大多数 step 没有可用梯度。`logs/train-OpenR1/grpo.log` 里，正确奖励全程大约 0.06–0.12，没有抬升。深层原因是 OpenR1 的题对 0.6B 太难。一组里经常全错，正确奖励出不来。

  继续优化：

1. GRPO 训练集改成 GSM8K。



## 优化方向 2 的第二轮

相对上一轮做了三处改动：SFT `max_length` 从 512 调到 4096；思维链里的 `\boxed{...}` 拆成普通文本，最终答案只留在 `<answer>`；GRPO 训练集从 OpenR1 换成 GSM8K。评测仍是 GSM8K test 前 20 题、贪心解码、`max_new_tokens=4096`。基座输出与上一轮逐字相同。明细见 `data/eval/OpenR1-finetuned-model-v1/`。

结论：`<answer>` **比例从约 45% 升到 75%–80%，指令遵循效果显著提升。且仔细比对发现，这个75%-80%等于回答完成率，即在4096个token内回答完成没有被截断的回答，其实100%按照指定格式来作答的。答对的题数，SFT 正确 13 / 20，GRPO 正确 11 / 20，都有提升且高于基座模型。但答对题数，这轮 GRPO 仍低于这轮 SFT。**


| 实验               | 模型                     | 正确      | 准确率     | 使用 `<answer>` | 使用 `\boxed{}` | 平均生成 token | 回答完成率        |
| ---------------- | ---------------------- | ------- | ------- | ------------- | ------------- | ---------- | ------------ |
| OpenR1 · 4096    | 原模型 Qwen3-0.6B         | 10 / 20 | 50%     | 10%           | 55%           | 2264       | 13 / 20（65%） |
| OpenR1 · 4096    | SFT LoRA               | 11 / 20 | 55%     | 45%           | 30%           | 1838       | 15 / 20（75%） |
| OpenR1 · 4096    | SFT merged + GRPO LoRA | 10 / 20 | 50%     | 45%           | 30%           | 1989       | 15 / 20（75%） |
| OpenR1 v1 · 4096 | 原模型 Qwen3-0.6B         | 10 / 20 | 50%     | 10%           | 55%           | 2264       | 13 / 20（65%） |
| OpenR1 v1 · 4096 | SFT LoRA               | 13 / 20 | **65%** | **80%**       | 0%            | 1736       | 16 / 20（80%） |
| OpenR1 v1 · 4096 | SFT merged + GRPO LoRA | 11 / 20 | 55%     | 75%           | 0%            | 1782       | 15 / 20（75%） |


- SFT 写完的 16 题全部使用 `<answer>`，没有再落到 `\boxed{}`。正确数从 11 增到 13：新做对第 12、14、16、18 题，第 15、19 题改为写满 4096。平均长度 1736 token，回答完成率 16 / 20（80%）。
- GRPO 的 `<answer>` 比例是 75%，`\boxed{}` 为 0。正确 11 / 20，比这轮 SFT 少 2 题（第 0 题 18 写成 38，第 12 题写满 4096）。相对上一轮 GRPO 的 10 / 20 多 1 题，完成率仍是 15 / 20（75%）。

