"""Compare GSM8K solve rate: base Qwen3, SFT LoRA, GRPO LoRA.

Uses the official test split (not SFT train[:500] or GRPO train[500:1000]).
Same prompt for all three. Predictions are taken from <answer>, then \\boxed{}.

Example (server):

CUDA_VISIBLE_DEVICES=2 python scripts/eval_math.py \
  --dataset data/raw/openai/gsm8k \
  --split test \
  --limit 20 \
  --base_model /rainbow/liuxm/learn/qwen3-0.6B \
  --sft_adapter outputs/sft/qwen3-0.6b-lora-math-format \
  --grpo_base outputs/sft/qwen3-0.6b-merged-math-format \
  --grpo_adapter outputs/grpo/qwen3-0.6b-gsm8k-from-sft \
  --output_dir data/eval
"""
import argparse
import gc
import json
import re
import sys
from pathlib import Path

import torch
from datasets import load_dataset
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gpu_utils import print_gpu_memory

MATH_PROMPT_TEMPLATE = (
    "Solve the following math problem. "
    "Put the final answer inside <answer></answer>.\n\n"
    "{question}"
)


def extract_gsm8k_gold(answer_text):
    if "####" in answer_text:
        return answer_text.split("####")[-1].strip().replace(",", "")
    return answer_text.strip().replace(",", "")


def extract_boxed(text):
    idx = text.rfind("\\boxed{")
    if idx == -1:
        return ""
    start = idx + len("\\boxed{")
    depth = 1
    i = start
    while i < len(text) and depth > 0:
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return text[start:i].strip()
        i += 1
    return text[start:].strip()


def extract_pred(text):
    matches = re.findall(r"<answer>(.*?)</answer>", text, re.DOTALL)
    if matches:
        return matches[-1].strip(), "answer_tag"
    boxed = extract_boxed(text)
    if boxed:
        return boxed, "boxed"
    if "####" in text:
        return text.split("####")[-1].strip().split("\n")[0].replace(",", ""), "hash"
    return "", "none"


def normalize_num(text):
    s = str(text).replace(",", "").replace("$", "").strip()
    s = s.replace(" ", "")
    try:
        value = float(s)
        if value.is_integer():
            return str(int(value))
        return str(value)
    except ValueError:
        return s


def load_rows(dataset_path, split, offset, limit):
    path = Path(dataset_path)
    if path.suffix == ".jsonl":
        rows = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                obj = json.loads(line)
                if "raw_question" in obj:
                    question = obj["raw_question"]
                    gold = str(obj["answer"])
                elif "question" in obj:
                    question = obj["question"]
                    gold = extract_gsm8k_gold(obj["answer"])
                else:
                    raise ValueError(f"Unrecognized jsonl schema: {obj.keys()}")
                rows.append({"question": question, "gold": gold})
        rows = rows[offset:]
        if limit is not None:
            rows = rows[:limit]
        return rows

    ds = load_dataset(str(path), split=split)
    end = len(ds) if limit is None else min(offset + limit, len(ds))
    rows = []
    for i in range(offset, end):
        ex = ds[i]
        rows.append({
            "question": ex["question"],
            "gold": extract_gsm8k_gold(ex["answer"]),
        })
    return rows


def load_model(base_model, adapter_path=None):
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    if adapter_path:
        model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()
    return tokenizer, model


def unload(model, tokenizer):
    del model
    del tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def generate_one(tokenizer, model, question, max_new_tokens):
    messages = [{"role": "user", "content": MATH_PROMPT_TEMPLATE.format(question=question)}]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True,
    )
    device = next(model.parameters()).device
    inputs = tokenizer(text, return_tensors="pt").to(device)
    with torch.inference_mode():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )
    gen_ids = out[0][inputs["input_ids"].shape[1]:]
    completion = tokenizer.decode(gen_ids, skip_special_tokens=True)
    return completion, int(gen_ids.shape[0])


def eval_model(name, tokenizer, model, rows, max_new_tokens):
    records = []
    n_correct = 0
    n_answer_tag = 0
    n_boxed = 0
    token_sum = 0

    for i, row in enumerate(rows, start=1):
        completion, n_tokens = generate_one(
            tokenizer, model, row["question"], max_new_tokens
        )
        pred_raw, src = extract_pred(completion)
        pred = normalize_num(pred_raw)
        gold = normalize_num(row["gold"])
        correct = pred == gold and pred != ""
        n_correct += int(correct)
        n_answer_tag += int(src == "answer_tag")
        n_boxed += int(src == "boxed")
        token_sum += n_tokens
        records.append({
            "index": i - 1,
            "question": row["question"],
            "gold": gold,
            "pred": pred,
            "pred_source": src,
            "correct": correct,
            "n_tokens": n_tokens,
            "completion": completion,
        })
        mark = "OK" if correct else "NO"
        print(
            f"[{name} {i}/{len(rows)}] {mark} gold={gold} pred={pred or '-'} "
            f"src={src} tokens={n_tokens}",
            flush=True,
        )

    n = len(rows) or 1
    summary = {
        "name": name,
        "n": len(rows),
        "correct": n_correct,
        "accuracy": n_correct / n,
        "answer_tag_rate": n_answer_tag / n,
        "boxed_rate": n_boxed / n,
        "mean_gen_tokens": token_sum / n,
    }
    return summary, records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="data/raw/openai/gsm8k")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--max_new_tokens", type=int, default=2048)
    parser.add_argument("--output_dir", type=str, default="data/eval")
    parser.add_argument("--base_model", type=str, default="/rainbow/liuxm/learn/qwen3-0.6B")
    parser.add_argument(
        "--sft_adapter",
        type=str,
        default="outputs/sft/qwen3-0.6b-lora-math-format",
    )
    parser.add_argument(
        "--grpo_base",
        type=str,
        default="outputs/sft/qwen3-0.6b-merged-math-format",
    )
    parser.add_argument(
        "--grpo_adapter",
        type=str,
        default="outputs/grpo/qwen3-0.6b-gsm8k-from-sft",
    )
    parser.add_argument(
        "--models",
        type=str,
        default="base,sft,grpo",
        help="Comma list: base,sft,grpo",
    )
    args = parser.parse_args()

    repo = Path(__file__).resolve().parents[1]
    dataset_path = args.dataset
    if not Path(dataset_path).is_absolute():
        dataset_path = str(repo / dataset_path)

    limit = None if args.limit < 0 else args.limit
    rows = load_rows(dataset_path, args.split, args.offset, limit)
    print(f"Loaded {len(rows)} examples from {dataset_path} split={args.split}", flush=True)
    if not rows:
        raise SystemExit("No eval examples loaded.")

    out_dir = Path(args.output_dir)
    if not out_dir.is_absolute():
        out_dir = repo / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    specs = {
        "base": {"base": args.base_model, "adapter": None},
        "sft": {"base": args.base_model, "adapter": args.sft_adapter},
        "grpo": {"base": args.grpo_base, "adapter": args.grpo_adapter},
    }
    wanted = [m.strip() for m in args.models.split(",") if m.strip()]
    for name in wanted:
        if name not in specs:
            raise SystemExit(f"Unknown model {name}. Use base,sft,grpo")

    summaries = []
    for name in wanted:
        spec = specs[name]
        adapter = spec["adapter"]
        print_gpu_memory(f"Before {name}:")
        print(f"Loading {name}: base={spec['base']} adapter={adapter}", flush=True)
        tokenizer, model = load_model(spec["base"], adapter)
        summary, records = eval_model(
            name, tokenizer, model, rows, args.max_new_tokens
        )
        summaries.append(summary)
        pred_path = out_dir / f"{name}_preds.jsonl"
        with pred_path.open("w", encoding="utf-8") as f:
            for rec in records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"Wrote {pred_path}", flush=True)
        print(
            f"{name}: acc={summary['accuracy']:.3f} "
            f"({summary['correct']}/{summary['n']}) "
            f"answer_tag={summary['answer_tag_rate']:.3f} "
            f"boxed={summary['boxed_rate']:.3f} "
            f"mean_tokens={summary['mean_gen_tokens']:.1f}",
            flush=True,
        )
        unload(model, tokenizer)
        print_gpu_memory(f"After unload {name}:")

    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summaries, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n===== summary =====")
    print(f"{'model':<8} {'n':>5} {'acc':>8} {'<answer>':>10} {'boxed':>8} {'tok':>8}")
    for s in summaries:
        print(
            f"{s['name']:<8} {s['n']:>5} {s['accuracy']:>8.3f} "
            f"{s['answer_tag_rate']:>10.3f} {s['boxed_rate']:>8.3f} "
            f"{s['mean_gen_tokens']:>8.1f}"
        )
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
