import argparse
import re
import sys
from pathlib import Path

import yaml

from datasets import load_dataset
from peft import LoraConfig
from trl import GRPOConfig, GRPOTrainer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gpu_utils import print_gpu_memory


def _resolve_deepspeed(cfg):
    path = cfg.get("deepspeed")
    if not path:
        return None
    resolved = Path(path)
    if not resolved.is_absolute():
        resolved = Path(__file__).resolve().parents[1] / path
    return str(resolved)


def extract_answer(text):
    matches = re.findall(r"<answer>(.*?)</answer>", text, re.DOTALL)
    if matches:
        return matches[-1].strip()
    return ""


def normalize_num(text):
    s = str(text).replace(",", "").replace("$", "").strip()
    try:
        value = float(s)
        if value.is_integer():
            return str(int(value))
        return str(value)
    except ValueError:
        return s


def format_reward(completions, **kwargs):
    rewards = []
    for completion in completions:
        text = completion[0]["content"] if isinstance(completion, list) else str(completion)
        has_format = ("<answer>" in text) and ("</answer>" in text)
        rewards.append(0.5 if has_format else 0.0)
    return rewards


def correctness_reward(completions, answer=None, **kwargs):
    rewards = []
    for completion, gold in zip(completions, answer):
        text = completion[0]["content"] if isinstance(completion, list) else str(completion)
        pred = normalize_num(extract_answer(text))
        gold = normalize_num(gold)
        rewards.append(1.0 if pred == gold and pred != "" else 0.0)
    return rewards


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    print("Loading local processed GRPO dataset...")
    dataset = load_dataset("json", data_files=cfg["dataset_path"], split="train")
    print(dataset)
    print(dataset[0])

    peft_config = LoraConfig(
        r=cfg["lora_r"],
        lora_alpha=cfg["lora_alpha"],
        lora_dropout=cfg["lora_dropout"],
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ],
    )

    training_args = GRPOConfig(
        output_dir=cfg["output_dir"],
        max_steps=cfg["max_steps"],
        per_device_train_batch_size=cfg["per_device_train_batch_size"],
        gradient_accumulation_steps=cfg["gradient_accumulation_steps"],
        learning_rate=cfg["learning_rate"],
        logging_steps=cfg["logging_steps"],
        save_steps=cfg["save_steps"],
        max_completion_length=cfg["max_completion_length"],
        num_generations=cfg["num_generations"],
        num_iterations=cfg.get("num_iterations", 1),
        temperature=cfg.get("temperature", 0.8),
        warmup_steps=cfg.get("warmup_steps", 0),
        chat_template_kwargs={"enable_thinking": True},
        gradient_checkpointing=True,
        torch_empty_cache_steps=cfg.get("torch_empty_cache_steps", 1),
        use_liger_kernel=True,
        bf16=True,
        report_to="none",
        deepspeed=_resolve_deepspeed(cfg),
    )

    print_gpu_memory("Before trainer:")

    trainer = GRPOTrainer(
        model=cfg["model_name"],
        reward_funcs=[format_reward, correctness_reward],
        args=training_args,
        train_dataset=dataset,
        peft_config=peft_config,
    )

    print_gpu_memory("After trainer:")
    trainer.train()
    print_gpu_memory("After training:")

    trainer.save_model(cfg["output_dir"])
    print(f"Saved GRPO LoRA adapter to {cfg['output_dir']}")


if __name__ == "__main__":
    main()
    # 1) merge SFT LoRA → outputs/sft/qwen3-0.6b-merged-math-format
    # 2) train GRPO:
    # 单卡:
    # CUDA_VISIBLE_DEVICES=2 python scripts/train_grpo.py --config configs/grpo_qwen3_0.6b_gsm8k.yaml
    #
    # A40 卡1+卡2（不要再加 --use_deepspeed/--deepspeed_config_file，json 由 GRPOConfig.deepspeed 注入）:
    # CUDA_VISIBLE_DEVICES=1,2 accelerate launch \
    #   --num_processes 2 --mixed_precision bf16 \
    #   scripts/train_grpo.py --config configs/grpo_qwen3_0.6b_gsm8k.yaml
