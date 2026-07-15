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
