"""
数据准备脚本

注意SFT训练集要对齐 Qwen3 chat template，<think> 推理 + <answer> 最终答案

"""
import json
import os
import re
from datasets import load_dataset

WORK_DIR = '/Users/liuxm/IdeaProjects/LLM_Train/trl-post-training/'
RAW_DATA_DIR = WORK_DIR + 'data/raw/'
PREVIEW_DIR = WORK_DIR + 'data/sample_preview'
PROCESSED_DIR = WORK_DIR + 'data/processed'

os.makedirs(PREVIEW_DIR, exist_ok=True)
os.makedirs(os.path.join(PROCESSED_DIR, 'sft'), exist_ok=True)
os.makedirs(os.path.join(PROCESSED_DIR, 'grpo'), exist_ok=True)

MATH_PROMPT_TEMPLATE = (
    "Solve the following math problem. "
    "Put the final answer inside <answer></answer>.\n\n"
    "{question}"
)


def save_json(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def save_jsonl(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def preview_dataset(name, dataset, n=3):
    print(f"\n===== {name} =====")
    print(dataset)
    print("columns:", dataset.column_names)
    print("sample:", dataset[0])
    samples = [dataset[i] for i in range(min(n, len(dataset)))]
    save_json(os.path.join(PREVIEW_DIR, f"{name}_sample.json"), samples)


def extract_gsm8k_gold(answer_text):
    """GSM8K answer 字段格式: 'solution text\n#### 42'"""
    if "####" in answer_text:
        return answer_text.split("####")[-1].strip().replace(",", "")
    return answer_text.strip().replace(",", "")


def extract_gsm8k_solution(answer_text):
    """提取 GSM8K 的解题过程（不含 #### 后的最终答案）"""
    if "####" in answer_text:
        return answer_text.split("####")[0].strip()
    return answer_text.strip()


# GSM8K 把计算器中间结果写成 <<48/2=24>>，SFT 会把这种标注学成胡乱算式
GSM8K_CALC_RE = re.compile(r"<<[^>]*>>")


def clean_gsm8k_solution(answer_text):
    """去掉 GSM8K 的 <<expr=value>> 标注，保留书面计算过程。"""
    return GSM8K_CALC_RE.sub("", extract_gsm8k_solution(answer_text)).strip()


def format_math_completion(solution, gold):
    """Qwen3 目标格式：推理进 <think>，可见回复只有 <answer>。

    Qwen3 chat template 在 assistant 没有 </think> 时会插入空的
    <think></think>，SFT 就会学会跳过推理、也不稳定输出 <answer>。
    """
    return f"<think>\n{solution}\n</think>\n<answer>{gold}</answer>"


def prepare_sft(max_items=500):
    """SFT 数据：GSM8K 改造为 messages 格式。

    assistant = <think>解题过程</think><answer>gold</answer>
    训练后用作 GRPO 起点（reward 从 <answer> 抽取）。
    """
    ds = load_dataset(RAW_DATA_DIR + 'openai/gsm8k', split='train')
    preview_dataset("gsm8k", ds)

    rows = []
    for ex in ds.select(range(min(max_items, len(ds)))):
        gold = extract_gsm8k_gold(ex["answer"])
        solution = clean_gsm8k_solution(ex["answer"])
        rows.append({
            "messages": [
                {"role": "user", "content": MATH_PROMPT_TEMPLATE.format(question=ex["question"])},
                {"role": "assistant", "content": format_math_completion(solution, gold)},
            ]
        })

    out = os.path.join(PROCESSED_DIR, 'sft', 'math_format_500.jsonl')
    save_jsonl(out, rows)
    print(f"saved processed SFT subset ({len(rows)} rows): {out}")


def prepare_grpo(max_items=500, sft_offset=500):
    """GRPO 数据：GSM8K holdout，题难度与 SFT 同级。

    使用 train[sft_offset:sft_offset+max_items]，默认跳过 SFT 用过的前 500 条，
    避免 GRPO 只是把 SFT 训练集再强化一遍。
    """
    ds = load_dataset(RAW_DATA_DIR + 'openai/gsm8k', split='train')
    preview_dataset("gsm8k_for_grpo", ds)

    start = min(sft_offset, len(ds))
    end = min(start + max_items, len(ds))
    if start >= end:
        raise ValueError(f"GSM8K train 不够切 GRPO holdout: len={len(ds)}, offset={sft_offset}")

    rows = []
    for ex in ds.select(range(start, end)):
        gold = extract_gsm8k_gold(ex["answer"])
        prompt = MATH_PROMPT_TEMPLATE.format(question=ex["question"])
        rows.append({
            "prompt": [{"role": "user", "content": prompt}],
            "answer": gold,
            "raw_question": ex["question"],
            "raw_answer": ex["answer"],
        })

    out = os.path.join(PROCESSED_DIR, 'grpo', 'gsm8k_500.jsonl')
    save_jsonl(out, rows)
    print(f"saved processed GRPO subset ({len(rows)} rows, train[{start}:{end}]): {out}")


if __name__ == "__main__":
    prepare_sft()
    prepare_grpo()
