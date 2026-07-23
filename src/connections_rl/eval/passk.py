"""Best-of-k sampling eval: capability under sampling, not just greedy decoding.

Distinguishes capability destruction from greedy-decoding degradation: an arm
whose greedy output fails but whose pass@16 is high retains the capability in
its distribution; an arm near-zero at pass@16 has lost it.

Run (against a vLLM server hosting the arms in the eval config):
    python -m connections_rl.eval.passk --config configs/eval/qwen7b.yaml \
        --k 16 --temperature 0.9 --out results-analysis/passk-7b.json
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from connections_rl.data.formatting import build_chat
from connections_rl.data.loader import Puzzle, load_puzzles, normalize_word
from connections_rl.eval.stats import bootstrap_ci
from connections_rl.reward.reward import reward_breakdown


def evaluate_passk(
    model: str,
    puzzles: list[Puzzle],
    k: int,
    temperature: float,
    max_tokens: int = 512,
    base_url: str | None = None,
) -> list[dict]:
    from openai import OpenAI  # deferred: heavy optional dep

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
                n=k,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            texts = [c.message.content or "" for c in resp.choices]
        except Exception as e:  # endpoint hiccup -> counts as all-invalid
            print(f"[passk] {model} puzzle {p.puzzle_id}: {e}")
            texts = []
        bds = [reward_breakdown(t, list(p.words), answer_sets) for t in texts]
        records.append(
            {
                "puzzle_id": p.puzzle_id,
                "date": p.date,
                "any_solved": any(b.solved for b in bds),
                "any_valid": any(b.valid for b in bds),
                "max_groups_correct": max((b.correct_groups for b in bds), default=0),
                "best_reward": max((b.total for b in bds), default=None),
                "n_samples": len(bds),
            }
        )
    return records


def summarize(records: list[dict], k: int, n_resamples: int = 1000) -> dict:
    solves = [1.0 if r["any_solved"] else 0.0 for r in records]
    valid = [1.0 if r["any_valid"] else 0.0 for r in records]
    groups = [r["max_groups_correct"] / 4 for r in records]
    return {
        "k": k,
        "n": len(records),
        "pass_at_k_solve": bootstrap_ci(solves, n_resamples=n_resamples),
        "pass_at_k_valid": bootstrap_ci(valid, n_resamples=n_resamples),
        "best_of_k_groups_correct": bootstrap_ci(groups, n_resamples=n_resamples),
    }


def main(argv: list[str] | None = None) -> None:
    import yaml

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True, help="eval config listing the served arms")
    ap.add_argument("--k", type=int, default=16)
    ap.add_argument("--temperature", type=float, default=0.9)
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    cfg = yaml.safe_load(Path(args.config).read_text())
    puzzles = load_puzzles(cfg["puzzles"])
    out: dict = {"k": args.k, "temperature": args.temperature, "arms": {}}
    for arm in cfg["arms"]:
        records = evaluate_passk(arm["model"], puzzles, args.k, args.temperature, args.max_tokens)
        summary = summarize(records, args.k)
        out["arms"][arm["name"]] = {"summary": summary, "records": records}
        s = summary
        print(
            f"{arm['name']:>8s}: pass@{args.k} solve={s['pass_at_k_solve'][0]:.3f} "
            f"[{s['pass_at_k_solve'][1]:.3f},{s['pass_at_k_solve'][2]:.3f}]  "
            f"best-of-k groups={s['best_of_k_groups_correct'][0]:.3f}  "
            f"any-valid={s['pass_at_k_valid'][0]:.3f}"
        )
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=1))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
