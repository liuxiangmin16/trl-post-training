"""
OpenR1-Math 数据准备。

SFT 用 math_verify 判对的长 CoT，对齐 Qwen3 chat template：
assistant = <think>推理</think><answer>gold</answer>
GRPO 用同一筛选池里 SFT 之后的题目，只保留题面和数字答案。

先各取 500 条，方便和上一轮 GSM8K 短解实验对比。
推理按字符数卡在可训练区间，避免 </think><answer> 在 max_length 里被截掉。
当前 SFT max_length / GRPO max_completion_length 仍是 512，训这批数据前要调到至少 2048。
"""
import json
import os
import re
from pathlib import Path

from datasets import load_dataset

ROOT = Path(__file__).resolve().parents[1]
RAW_OPENR1_DIR = ROOT / "data" / "raw" / "OpenR1"
PREVIEW_DIR = ROOT / "data" / "sample_preview"
PROCESSED_DIR = ROOT / "data" / "processed"

SFT_MAX_ITEMS = 500
GRPO_MAX_ITEMS = 500
# GSM8K 金标解大约一两百字符。下限保证这批是长推理；
# 上限大约对应 2048 token 里还能放下 prompt 和 <answer>。
MIN_THINK_CHARS = 800
MAX_THINK_CHARS = 4000

os.makedirs(PREVIEW_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR / "sft", exist_ok=True)
os.makedirs(PROCESSED_DIR / "grpo", exist_ok=True)

MATH_PROMPT_TEMPLATE = (
    "Solve the following math problem. "
    "Put the final answer inside <answer></answer>.\n\n"
    "{question}"
)

THINK_RE = re.compile(r"<think>\s*(.*?)\s*</think>", re.DOTALL)
NUMERIC_RE = re.compile(r"-?\d+(?:\.\d+)?")


def save_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def save_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def normalize_numeric_answer(answer):
    """收成和 GRPO correctness_reward 一致的数字字符串。对不上就返回 None。"""
    s = str(answer).strip().replace(",", "").replace("$", "")
    s = re.sub(r"\s+", "", s)
    if not NUMERIC_RE.fullmatch(s):
        return None
    value = float(s)
    if value.is_integer():
        return str(int(value))
    return str(value)


def pick_reasoning(example):
    """在判对的 generation 里，取落在长度区间内最长的 <think> 正文。"""
    flags = example.get("correctness_math_verify") or []
    generations = example.get("generations") or []
    best = None
    for generation, ok in zip(generations, flags):
        if not ok or not isinstance(generation, str):
            continue
        match = THINK_RE.search(generation)
        if not match:
            continue
        body = match.group(1).strip()
        length = len(body)
        if length < MIN_THINK_CHARS or length > MAX_THINK_CHARS:
            continue
        if best is None or length > best[0]:
            best = (length, body)
    return None if best is None else best[1]


def iter_openr1():
    files = sorted(RAW_OPENR1_DIR.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"OpenR1 parquet 不存在: {RAW_OPENR1_DIR}")
    for path in files:
        dataset = load_dataset("parquet", data_files=str(path), split="train")
        for example in dataset:
            yield example


def format_math_completion(solution, gold):
    """Qwen3 目标格式：推理进 <think>，可见回复只有 <answer>。

    Qwen3 chat template 在 assistant 没有 </think> 时会插入空的
    <think></think>，SFT 就会学会跳过推理、也不稳定输出 <answer>。
    """
    return f"<think>\n{solution}\n</think>\n<answer>{gold}</answer>"


def collect_eligible(max_items):
    """按文件顺序收集可用题，凑够就停。

    可用：答案是纯数字（现有 reward 能打分），且有一条判对的长 CoT。
    """
    rows = []
    seen = set()
    scanned = 0
    for example in iter_openr1():
        scanned += 1
        gold = normalize_numeric_answer(example.get("answer"))
        problem = (example.get("problem") or "").strip()
        uuid = example.get("uuid") or problem
        if not gold or not problem or uuid in seen:
            continue
        reasoning = pick_reasoning(example)
        if reasoning is None:
            continue
        seen.add(uuid)
        prompt = MATH_PROMPT_TEMPLATE.format(question=problem)
        rows.append({
            "uuid": uuid,
            "prompt": prompt,
            "problem": problem,
            "answer": gold,
            "reasoning": reasoning,
        })
        if len(rows) >= max_items:
            break
    if len(rows) < max_items:
        raise RuntimeError(
            f"OpenR1 可用题只有 {len(rows)}，不够 {max_items} "
            f"（已扫描 {scanned} 条）"
        )
    print(f"scanned {scanned} rows, kept {len(rows)} eligible")
    return rows


def prepare_openr1(sft_items=SFT_MAX_ITEMS, grpo_items=GRPO_MAX_ITEMS):
    eligible = collect_eligible(sft_items + grpo_items)
    sft_rows_src = eligible[:sft_items]
    grpo_rows_src = eligible[sft_items:sft_items + grpo_items]

    sft_rows = []
    for row in sft_rows_src:
        sft_rows.append({
            "messages": [
                {"role": "user", "content": row["prompt"]},
                {"role": "assistant", "content": format_math_completion(row["reasoning"], row["answer"])},
            ]
        })

    grpo_rows = []
    for row in grpo_rows_src:
        grpo_rows.append({
            "prompt": [{"role": "user", "content": row["prompt"]}],
            "answer": row["answer"],
            "raw_question": row["problem"],
            "raw_answer": row["answer"],
        })

    sft_out = PROCESSED_DIR / "sft" / "openr1_math_500.jsonl"
    grpo_out = PROCESSED_DIR / "grpo" / "openr1_math_500.jsonl"
    save_jsonl(sft_out, sft_rows)
    save_jsonl(grpo_out, grpo_rows)
    save_json(PREVIEW_DIR / "openr1_sft_sample.json", sft_rows[:2])
    save_json(PREVIEW_DIR / "openr1_grpo_sample.json", grpo_rows[:2])

    think_lens = [len(row["reasoning"]) for row in sft_rows_src]
    think_lens.sort()
    print(f"saved SFT ({len(sft_rows)} rows): {sft_out}")
    print(f"saved GRPO ({len(grpo_rows)} rows): {grpo_out}")
    print(
        "SFT think chars: "
        f"min={think_lens[0]} median={think_lens[len(think_lens) // 2]} max={think_lens[-1]}"
    )


if __name__ == "__main__":
    prepare_openr1()
