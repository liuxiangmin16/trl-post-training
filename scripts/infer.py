import argparse
import sys
from pathlib import Path
from threading import Thread

import torch

from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
from peft import PeftModel

sys.path.insert(0, str(Path(__file__).resolve().parent))

# 与prepare_data.py 共用，推理 prompt 必须和训练一致
MATH_PROMPT_TEMPLATE = (
    "Solve the following math problem. "
    "Put the final answer inside <answer></answer>.\n\n"
    "{question}"
)
MATH_PROMPT_MARKER = "Put the final answer inside <answer></answer>."


def load_model(base_model, adapter_path=None):
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)

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


def wrap_math_prompt(prompt):
    if MATH_PROMPT_MARKER in prompt:
        return prompt
    return MATH_PROMPT_TEMPLATE.format(question=prompt)


def generate(tokenizer, model, prompt, temperature=0.0, top_p=0.9, seed=None):
    messages = [{"role": "user", "content": wrap_math_prompt(prompt)}]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True,
    )

    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    streamer = TextIteratorStreamer(
        tokenizer, skip_prompt=True, skip_special_tokens=True
    )

    do_sample = temperature > 0
    generation_kwargs = dict(
        **inputs,
        max_new_tokens=2048,
        do_sample=do_sample,
        streamer=streamer,
    )
    if do_sample:
        generation_kwargs["temperature"] = temperature
        generation_kwargs["top_p"] = top_p
        if seed is not None:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()

    for text in streamer:
        print(text, end="", flush=True)

    thread.join()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", type=str, default="Qwen/Qwen3-0.6B")
    parser.add_argument("--adapter", type=str, default=None)
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="0 = greedy (same every run). >0 enables sampling, e.g. 0.7",
    )
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    tokenizer, model = load_model(args.base_model, args.adapter)
    generate(
        tokenizer,
        model,
        args.prompt,
        temperature=args.temperature,
        top_p=args.top_p,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()