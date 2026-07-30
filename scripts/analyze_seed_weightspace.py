"""Weight-space convergence of independent GRPO seeds.

Tests whether independently seeded GRPO runs move the policy in the SAME
direction (a systematic attractor of the reward) or in unrelated directions
(seed noise). Compares the RL-induced effective LoRA update

    dW_RL(seed) = B_grpo @ A_grpo - B_sft @ A_sft

across seeds, per module and in aggregate. All inner products are computed in
the low-rank factors, so dense dW (hundreds of MB per module at 7B) is never
materialized:

    <B1 A1, B2 A2>_F = tr( (B1^T B2)(A2 A1^T) )     # r x r matrices only

Usage (adapter .safetensors files downloaded to a directory):
    python scripts/analyze_seed_weightspace.py --dir /tmp/adp \
        --sft connections-rl-sft-7b.safetensors \
        --seeds seed0=connections-rl-grpo-7b.safetensors \
                seed1=connections-rl-grpo-7b-seed1.safetensors \
                seed2=connections-rl-grpo-7b-seed2.safetensors
"""

from __future__ import annotations

import argparse
import itertools
import json
import struct
from pathlib import Path

import numpy as np


def _read_header(path: Path) -> tuple[dict, int]:
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        return json.loads(f.read(n)), 8 + n


def _read_tensor(path: Path, hdr: dict, base: int, key: str) -> np.ndarray:
    meta = hdr[key]
    start, end = meta["data_offsets"]
    with open(path, "rb") as f:
        f.seek(base + start)
        raw = f.read(end - start)
    dtype = meta["dtype"]
    if dtype == "F32":
        arr = np.frombuffer(raw, dtype=np.float32)
    elif dtype == "F16":
        arr = np.frombuffer(raw, dtype=np.float16)
    elif dtype == "BF16":
        # bf16 is the high 16 bits of fp32, so widening is exact.
        u16 = np.frombuffer(raw, dtype=np.uint16).astype(np.uint32)
        arr = (u16 << 16).view(np.float32)
    else:
        raise ValueError(f"unsupported dtype {dtype} for {key}")
    return arr.reshape(meta["shape"]).astype(np.float64)


class Adapter:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.hdr, self.base = _read_header(path)

    @property
    def modules(self) -> list[str]:
        return sorted({k.rsplit(".lora_", 1)[0] for k in self.hdr if ".lora_" in k})

    def factors(self, module: str) -> tuple[np.ndarray, np.ndarray]:
        b = _read_tensor(self.path, self.hdr, self.base, module + ".lora_B.weight")
        a = _read_tensor(self.path, self.hdr, self.base, module + ".lora_A.weight")
        return b, a


def frob_ip(b1: np.ndarray, a1: np.ndarray, b2: np.ndarray, a2: np.ndarray) -> float:
    """<B1 A1, B2 A2>_F without forming the dense products."""
    return float(np.trace((b1.T @ b2) @ (a2 @ a1.T)))


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dir", required=True)
    ap.add_argument("--sft", required=True, help="SFT adapter filename (shared warm start)")
    ap.add_argument("--seeds", nargs="+", required=True, help="name=filename pairs")
    ap.add_argument("--out", help="optional path to write the report")
    args = ap.parse_args(argv)

    root = Path(args.dir)
    sft = Adapter(root / args.sft)
    seeds = {}
    for spec in args.seeds:
        name, _, fname = spec.partition("=")
        seeds[name] = Adapter(root / fname)
    mods = sft.modules
    names = list(seeds)

    lines: list[str] = []

    def emit(s: str = "") -> None:
        print(s)
        lines.append(s)

    emit(f"modules: {len(mods)}   seeds: {', '.join(names)}")
    emit("(effective LoRA updates compared via low-rank factors; dense dW never formed)")

    sft_sq = 0.0
    rl_sq = dict.fromkeys(names, 0.0)
    rl_ip_sft = dict.fromkeys(names, 0.0)
    pair_ip = {p: 0.0 for p in itertools.combinations(names, 2)}
    per_mod: list[tuple[str, float, float]] = []

    for m in mods:
        bs, as_ = sft.factors(m)
        sft_sq += frob_ip(bs, as_, bs, as_)
        # dW_RL = B_g A_g - B_s A_s is rank 2r: [B_g, -B_s] @ [A_g; A_s]
        rf = {}
        for name, ad in seeds.items():
            bg, ag = ad.factors(m)
            rf[name] = (np.concatenate([bg, -bs], axis=1), np.concatenate([ag, as_], axis=0))
        for name in names:
            b, a = rf[name]
            rl_sq[name] += frob_ip(b, a, b, a)
            rl_ip_sft[name] += frob_ip(b, a, bs, as_)
        for pair in pair_ip:
            pair_ip[pair] += frob_ip(*rf[pair[0]], *rf[pair[1]])
        if len(names) >= 2:
            b0, a0 = rf[names[0]]
            b1, a1 = rf[names[1]]
            n0 = frob_ip(b0, a0, b0, a0) ** 0.5
            n1 = frob_ip(b1, a1, b1, a1) ** 0.5
            per_mod.append((m, frob_ip(b0, a0, b1, a1) / (n0 * n1), n0))

    emit()
    emit("=== magnitude of RL-induced change (Frobenius, all modules) ===")
    emit(f"SFT update (base -> SFT)   ||dW|| = {sft_sq**0.5:.4f}")
    for name in names:
        mag = rl_sq[name] ** 0.5
        emit(f"{name}: RL update ||dW|| = {mag:.4f}   ({mag / sft_sq**0.5:.2f}x SFT update)")

    emit()
    emit("=== cosine similarity between independent seeds' RL-update directions ===")
    for (a, b), v in pair_ip.items():
        emit(f"cos({a}, {b}) = {v / ((rl_sq[a] * rl_sq[b]) ** 0.5):+.4f}")

    emit()
    emit("=== control: cos(RL update, SFT update) ===")
    for name in names:
        emit(f"cos({name}_RL, SFT) = {rl_ip_sft[name] / ((rl_sq[name] * sft_sq) ** 0.5):+.4f}")

    dim = sum(
        sft.hdr[m + ".lora_B.weight"]["shape"][0] * sft.hdr[m + ".lora_A.weight"]["shape"][1]
        for m in mods
    )
    rand = (2 / (np.pi * dim)) ** 0.5
    emit()
    emit(f"dW parameter dim = {dim:,};  E[|cos|] for random directions ~ {rand:.2e}")

    if per_mod:
        per_mod.sort(key=lambda t: -t[2])
        emit()
        emit(f"=== per-module cos({names[0]}, {names[1]}), 10 largest-change modules ===")
        for m, c, n in per_mod[:10]:
            short = m.replace("base_model.model.model.layers.", "L").replace(".weight", "")
            emit(f"  {short:<34} cos={c:+.3f}  ||dW||={n:.4f}")
        cs = np.array([c for _, c, _ in per_mod])
        emit()
        emit(
            f"per-module cos: median={np.median(cs):+.3f}  mean={cs.mean():+.3f}  "
            f"min={cs.min():+.3f}  max={cs.max():+.3f}"
        )

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text("\n".join(lines) + "\n")
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
