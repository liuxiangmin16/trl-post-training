import torch


def print_gpu_memory(prefix=""):
    import os

    if os.environ.get("LOCAL_RANK", "0") not in ("0", "-1"):
        return
    if not torch.cuda.is_available():
        print(f"{prefix} CUDA not available")
        return

    allocated = torch.cuda.memory_allocated() / 1024**3
    reserved = torch.cuda.memory_reserved() / 1024**3
    max_allocated = torch.cuda.max_memory_allocated() / 1024**3

    print(
        f"{prefix} GPU memory | "
        f"allocated={allocated:.2f}GB | "
        f"reserved={reserved:.2f}GB | "
        f"max_allocated={max_allocated:.2f}GB"
    )
