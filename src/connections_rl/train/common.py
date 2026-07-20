"""Shared training utilities: config loading, seeding, dataset prep.

Heavy libraries (torch/transformers/trl) are imported inside functions so the
core package imports cleanly on machines without them.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any


def load_config(path: str | Path) -> dict[str, Any]:
    import yaml

    cfg = yaml.safe_load(Path(path).read_text())
    # Model config can be referenced by path for reuse across sft/grpo.
    model_ref = cfg.get("model_config")
    if model_ref:
        cfg["model"] = yaml.safe_load(Path(model_ref).read_text())
    return cfg


def set_seed(seed: int) -> None:
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def fix_qlora_adapter_dtype_for_pre_ampere(model: Any) -> None:
    """Undo TRL's unconditional bf16 cast of QLoRA adapter params on pre-Ampere GPUs.

    Newer TRL casts all trainable params of quantized models to bf16 inside
    SFTTrainer/GRPOTrainer.__init__ (QLoRA-paper recommendation, no opt-out:
    huggingface/peft#2889). On pre-Ampere GPUs (T4) training runs under fp16
    AMP, whose GradScaler cannot unscale bf16 grads
    ("_amp_foreach_non_finite_check_and_unscale_cuda not implemented for
    BFloat16"). Re-cast trainable params to fp32 — the standard fp16-AMP
    master-weight dtype. Call AFTER trainer construction, BEFORE train()
    (the optimizer is created lazily in train(), so it picks up fp32 params).
    """
    import torch

    if not (torch.cuda.is_available() and torch.cuda.get_device_capability()[0] < 8):
        return
    n = 0
    for param in model.parameters():
        if param.requires_grad and param.dtype == torch.bfloat16:
            param.data = param.data.float()
            n += 1
    if n:
        print(f"[train] re-cast {n} bf16 trainable params to fp32 (pre-Ampere fp16 AMP)")


def read_jsonl(path: str | Path) -> list[dict]:
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                out.append(json.loads(line))
    return out


def load_puzzle_split(split_dir: str | Path, split: str) -> list[dict]:
    """Raw puzzle records for GRPO rollouts (puzzles_{split}.json from `make data`)."""
    return json.loads((Path(split_dir) / f"puzzles_{split}.json").read_text())
