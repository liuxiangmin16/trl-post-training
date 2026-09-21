import argparse
import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", type=str, required=True,
                        help="底座模型 ID 或本地路径，如 Qwen/Qwen3-0.6B")
    parser.add_argument("--lora_path", type=str, required=True,
                        help="LoRA 权重目录，如 outputs/sft/qwen3-0.6b-lora-math-format")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="合并后模型保存路径")
    args = parser.parse_args()

    print(f"Loading base model from {args.base_model} ...")
    base = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16,
        device_map="cpu",  # 合并用 CPU 即可，省显存
    )
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)

    print(f"Loading LoRA adapter from {args.lora_path} ...")
    model = PeftModel.from_pretrained(base, args.lora_path)

    print("Merging LoRA into base model ...")
    merged = model.merge_and_unload()  # 关键：把 LoRA 权重合并并卸载 peft 包装

    print(f"Saving merged model to {args.output_dir} ...")
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(args.output_dir, safe_serialization=True)
    tokenizer.save_pretrained(args.output_dir)

    print(f"Done. Merged model saved at: {args.output_dir}")


if __name__ == "__main__":
    main()

# python scripts/merge_lora.py \
#   --base_model /rainbow/liuxm/learn/qwen3-0.6B \
#   --lora_path outputs/sft/qwen3-0.6b-lora-math-format \
#   --output_dir outputs/sft/qwen3-0.6b-merged-math-format
#
# 再把 GRPO 的 model_name 改成 merged 目录后训练。