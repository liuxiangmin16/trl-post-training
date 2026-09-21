import argparse
import functools
import inspect
import types
import sys
from pathlib import Path

import yaml

from datasets import load_dataset
from peft import LoraConfig
from trl import SFTConfig, SFTTrainer

import trl.trainer.sft_trainer as _sft_mod

# ---- monkey-patch: 兼容 transformers 4.57+ 中 forward 为 functools.partial 的情况 ----
# trl 1.13.0 的 _patch_chunked_ce_lm_head 第 386 行调用 original_forward.__func__，
# 但 functools.partial 没有 __func__ 属性，导致 AttributeError。
# 方案：在调用原函数前，如果 model.forward 是 partial，先用 MethodType 包装一层，
#       使其拥有 __func__ 属性，原函数即可正常工作。
_orig_patch_chunked_ce_lm_head = _sft_mod._patch_chunked_ce_lm_head


def _patched_chunked_ce_lm_head(model, chunk_size, is_vlm=False):
    original_forward = model.forward
    if isinstance(original_forward, functools.partial):
        # 用 MethodType 包装 partial，使其表现得像 bound method（有 __func__）
        _partial = original_forward

        def _forward_wrapper(self, *args, **kwargs):
            return _partial(*args, **kwargs)

        model.forward = types.MethodType(_forward_wrapper, model)
    _orig_patch_chunked_ce_lm_head(model, chunk_size, is_vlm=is_vlm)


_sft_mod._patch_chunked_ce_lm_head = _patched_chunked_ce_lm_head
# ---------------------------------------------------------------------------------------

# sys.path.insert(0, str(Path(__file__).resolve().parent))
from gpu_utils import print_gpu_memory


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    print("Loading local processed SFT dataset...")
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

    training_args = SFTConfig(
        output_dir=cfg["output_dir"],
        max_steps=cfg["max_steps"],
        per_device_train_batch_size=cfg["per_device_train_batch_size"],
        gradient_accumulation_steps=cfg["gradient_accumulation_steps"],
        learning_rate=cfg["learning_rate"],
        max_length=cfg["max_length"],
        logging_steps=cfg["logging_steps"],
        save_steps=cfg["save_steps"],
        bf16=True,
        report_to="none",
        packing=False,
        # 只在 assistant 段（含 <think> / <answer> / <|im_end|>）上算 loss。
        # TRL 1.13 默认 False，会把 user prompt 也算进去，格式信号被稀释。
        assistant_only_loss=True,
    )

    print_gpu_memory("Before trainer:")

    trainer = SFTTrainer(
        model=cfg["model_name"],
        args=training_args,
        train_dataset=dataset,
        peft_config=peft_config,
    )

    print_gpu_memory("After trainer:")
    trainer.train()
    print_gpu_memory("After training:")

    trainer.save_model(cfg["output_dir"])
    print(f"Saved SFT LoRA adapter to {cfg['output_dir']}")


if __name__ == "__main__":
    main()
# CUDA_VISIBLE_DEVICES=2 python scripts/train_sft.py --config configs/sft_qwen3_0.6b_lora.yaml
