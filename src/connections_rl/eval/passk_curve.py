"""Full pass@k curve: unbiased estimator over a persisted per-sample pool.

Two subcommands, so generation (GPU/serving) and estimation (pure arithmetic)
are decoupled and the estimation is reproducible from committed data:

    # 1. Generate a pool of n samples per puzzle per arm, persisting EVERY
    #    per-sample outcome (this is what the k=16 pool did not do):
    python -m connections_rl.eval.passk_curve generate \
        --config configs/eval/qwen7b.yaml --n 32 --temperature 0.9 \
        --out results-analysis/passk-pool-7b-n32.json

    # 2. Compute the curve for k <= n from the pool (no GPU, deterministic):
    python -m connections_rl.eval.passk_curve curve \
        --pool results-analysis/passk-pool-7b-n32.json --k 1 2 4 8 16 32 \
        --out results-analysis/passk-curve-7b.json \
        --plot results-analysis/passk-curve-7b.png

Estimators (all exact, no Monte Carlo subsampling):

- Binary predicates (whole-puzzle solve, validity): the standard unbiased
  pass@k estimator from Chen et al. (Codex), per puzzle
      pass@k = 1 - C(n-c, k) / C(n, k)
  with n samples of which c succeed, averaged over puzzles. Computed via the
  stable running product 1 - prod_{i=0..k-1} (n-c-i)/(n-i) rather than large
  binomials.
- Graded score (groups correct, 0-4): the exact expectation of the maximum
  over a uniformly random k-subset of the n per-sample scores, via order
  statistics: with m_v = #samples with score <= v,
      P(max <= v) = C(m_v, k) / C(n, k)
  so E[max] = sum_v v * (P(max <= v) - P(max <= v-1)). This generalizes
  pass@k to per-group credit and reduces to it when scores are 0/1.

k > n is refused, never extrapolated. Success predicates are identical to the
greedy tables: `reward_breakdown` on the same parsed output.
"""

from __future__ import annotations

import argparse
import json
import os
from math import comb
from pathlib import Path

from connections_rl.eval.stats import bootstrap_ci

# --------------------------------------------------------------------------
# Estimators (pure arithmetic, unit-testable without a server)
# --------------------------------------------------------------------------


def pass_at_k(n: int, c: int, k: int) -> float:
    """Unbiased pass@k for one problem: n samples, c successes. Requires k <= n."""
    if k > n:
        raise ValueError(f"pass@k undefined for k={k} > n={n}; generate a larger pool")
    if c < 0 or c > n:
        raise ValueError(f"invalid success count c={c} for n={n}")
    if n - c < k:
        return 1.0
    prod = 1.0
    for i in range(k):
        prod *= (n - c - i) / (n - i)
    return 1.0 - prod


def expected_best_of_k(scores: list[float], k: int) -> float:
    """Exact E[max over a uniform random k-subset] of per-sample scores."""
    n = len(scores)
    if k > n:
        raise ValueError(f"best-of-k undefined for k={k} > n={n}")
    ordered = sorted(scores)
    total = comb(n, k)
    exp = 0.0
    prev_cdf = 0.0
    i = 0
    while i < n:
        j = i
        while j + 1 < n and ordered[j + 1] == ordered[i]:
            j += 1
        m = j + 1  # samples with score <= this value
        cdf = comb(m, k) / total if m >= k else 0.0
        exp += ordered[i] * (cdf - prev_cdf)
        prev_cdf = cdf
        i = j + 1
    return exp


# --------------------------------------------------------------------------
# Pool generation (mirrors eval/passk.py but persists per-sample outcomes)
# --------------------------------------------------------------------------


def generate_pool(
    model: str,
    puzzles: list,
    n: int,
    temperature: float,
    top_p: float,
    max_tokens: int,
    base_url: str | None = None,
) -> list[dict]:
    from openai import OpenAI  # deferred: heavy optional dep

    from connections_rl.data.formatting import build_chat
    from connections_rl.data.loader import normalize_word
    from connections_rl.reward.reward import reward_breakdown

    client = OpenAI(
        base_url=base_url or os.environ.get("CRL_BASE_URL", "http://localhost:8000/v1"),
        api_key=os.environ.get("CRL_API_KEY", "EMPTY"),
    )
    records = []
    for p in puzzles:
        answer_sets = [frozenset(normalize_word(w) for w in g.members) for g in p.groups]
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=build_chat(p),  # type: ignore[arg-type]
                n=n,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
            )
            texts = [c.message.content or "" for c in resp.choices]
        except Exception as e:  # endpoint hiccup -> recorded as zero samples
            print(f"[passk_curve] {model} puzzle {p.puzzle_id}: {e}")
            texts = []
        bds = [reward_breakdown(t, list(p.words), answer_sets) for t in texts]
        records.append(
            {
                "puzzle_id": p.puzzle_id,
                "date": p.date,
                "n_samples": len(bds),
                # per-sample outcomes -- the whole point of this pool format
                "solved": [bool(b.solved) for b in bds],
                "valid": [bool(b.valid) for b in bds],
                "groups_correct": [int(b.correct_groups) for b in bds],
                "reward": [float(b.total) for b in bds],
            }
        )
    return records


def cmd_generate(args: argparse.Namespace) -> None:
    import yaml

    from connections_rl.data.loader import load_puzzles

    cfg = yaml.safe_load(Path(args.config).read_text())
    puzzles = load_puzzles(cfg["puzzles"])
    vllm_version = os.environ.get("CRL_VLLM_VERSION", "unrecorded")
    out: dict = {
        "settings": {
            "n": args.n,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "max_tokens": args.max_tokens,
            "vllm_version": vllm_version,
            "note": "all arms generated in a single serving session; "
            "settings identical across arms by construction",
        },
        "arms": {},
    }
    for arm in cfg["arms"]:
        records = generate_pool(
            arm["model"], puzzles, args.n, args.temperature, args.top_p, args.max_tokens
        )
        short = sum(1 for r in records if r["n_samples"] < args.n)
        if short:
            print(f"[passk_curve] WARNING {arm['name']}: {short} puzzles with < n samples")
        out["arms"][arm["name"]] = records
        print(f"{arm['name']}: pool of {args.n} samples x {len(records)} puzzles done")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=1))
    print(f"wrote {args.out}")


# --------------------------------------------------------------------------
# Curve computation
# --------------------------------------------------------------------------


def cmd_curve(args: argparse.Namespace) -> None:
    pool = json.loads(Path(args.pool).read_text())
    settings = pool.get("settings", {})
    n_pool = settings.get("n")
    ks = sorted(set(args.k))
    curve: dict = {"settings": settings, "k": ks, "arms": {}}
    for arm, records in pool["arms"].items():
        usable = [r for r in records if r["n_samples"] > 0]
        dropped = len(records) - len(usable)
        if dropped:
            print(f"[passk_curve] WARNING {arm}: {dropped} puzzles dropped (0 samples)")
        arm_out: dict = {"n_puzzles": len(usable), "per_k": {}}
        for k in ks:
            bad = [r["puzzle_id"] for r in usable if k > r["n_samples"]]
            if bad:
                raise SystemExit(
                    f"k={k} exceeds n_samples for {len(bad)} puzzles in arm '{arm}' "
                    f"(e.g. {bad[:5]}). Refusing to extrapolate: regenerate a pool "
                    f"with n >= {k} or drop this k."
                )
            solve = [pass_at_k(r["n_samples"], sum(r["solved"]), k) for r in usable]
            valid = [pass_at_k(r["n_samples"], sum(r["valid"]), k) for r in usable]
            groups = [expected_best_of_k(r["groups_correct"], k) / 4 for r in usable]
            arm_out["per_k"][str(k)] = {
                "pass_at_k_solve": bootstrap_ci(solve),
                "pass_at_k_valid": bootstrap_ci(valid),
                "best_of_k_groups_correct_fraction": bootstrap_ci(groups),
            }
        curve["arms"][arm] = arm_out
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(curve, indent=1))
    print(f"wrote {args.out}  (estimator: unbiased pass@k / exact best-of-k, n={n_pool})")
    if args.plot:
        plot_curve(curve, args.plot)


def plot_curve(curve: dict, out_png: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ks = curve["k"]
    n = curve.get("settings", {}).get("n", "?")
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for arm, data in curve["arms"].items():
        mid = [data["per_k"][str(k)]["best_of_k_groups_correct_fraction"][0] * 100 for k in ks]
        lo = [data["per_k"][str(k)]["best_of_k_groups_correct_fraction"][1] * 100 for k in ks]
        hi = [data["per_k"][str(k)]["best_of_k_groups_correct_fraction"][2] * 100 for k in ks]
        ax.plot(ks, mid, marker="o", label=arm)
        ax.fill_between(ks, lo, hi, alpha=0.15)
    ax.set_xscale("log", base=2)
    ax.set_xticks(ks)
    ax.set_xticklabels([str(k) for k in ks])
    ax.set_xlabel("k (samples)")
    ax.set_ylabel("% of groups (best-of-k, exact estimator)")
    ax.set_title(f"pass@k / best-of-k groups, n={n} samples per puzzle, temp 0.9")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    print(f"wrote {out_png}")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("generate", help="generate a per-sample pool via a vLLM server")
    g.add_argument("--config", required=True)
    g.add_argument("--n", type=int, default=32)
    g.add_argument("--temperature", type=float, default=0.9)
    g.add_argument("--top-p", type=float, default=1.0)
    g.add_argument("--max-tokens", type=int, default=512)
    g.add_argument("--out", required=True)
    g.set_defaults(func=cmd_generate)

    c = sub.add_parser("curve", help="compute the pass@k curve from a persisted pool")
    c.add_argument("--pool", required=True)
    c.add_argument("--k", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32])
    c.add_argument("--out", required=True)
    c.add_argument("--plot", default=None)
    c.set_defaults(func=cmd_curve)

    args = ap.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
