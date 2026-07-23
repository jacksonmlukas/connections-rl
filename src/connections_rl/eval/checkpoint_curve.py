"""Structure-vs-semantics decomposition over GRPO training checkpoints.

Evaluates a sequence of served checkpoint adapters (greedy) on the VAL split,
recording the two reward components separately per checkpoint:
  * structural: fraction of outputs that are valid partitions of the board;
  * semantic:   mean fully-correct groups / 4.

Serve the checkpoints as vLLM LoRA modules first, then:
    python -m connections_rl.eval.checkpoint_curve \
        --arms "sft-7b:0,ckpt-50:50,ckpt-100:100,grpo-7b:403" \
        --puzzles data/splits/puzzles_val.json --out results-analysis/ckpt-7b

Uses the val split (not test) so the test set is never reused for analysis.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from connections_rl.data.formatting import build_chat
from connections_rl.data.loader import Puzzle, load_puzzles, normalize_word
from connections_rl.reward.reward import reward_breakdown


def evaluate_arm_components(
    model: str,
    puzzles: list[Puzzle],
    max_tokens: int = 512,
    base_url: str | None = None,
) -> dict:
    from openai import OpenAI  # deferred: heavy optional dep

    client = OpenAI(
        base_url=base_url or os.environ.get("CRL_BASE_URL", "http://localhost:8000/v1"),
        api_key=os.environ.get("CRL_API_KEY", "EMPTY"),
    )
    valid = 0
    groups_sum = 0.0
    reward_sum = 0.0
    solved = 0
    for p in puzzles:
        answer_sets = [frozenset(normalize_word(w) for w in g.members) for g in p.groups]
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=build_chat(p),  # type: ignore[arg-type]
                temperature=0.0,
                max_tokens=max_tokens,
            )
            text = resp.choices[0].message.content or ""
        except Exception as e:
            print(f"[ckpt] {model} puzzle {p.puzzle_id}: {e}")
            text = ""
        b = reward_breakdown(text, list(p.words), answer_sets)
        valid += int(b.valid)
        groups_sum += b.correct_groups / 4
        reward_sum += b.total
        solved += int(b.solved)
    n = len(puzzles)
    return {
        "n": n,
        "structural_valid_rate": valid / n,
        "semantic_groups_correct": groups_sum / n,
        "solve_rate": solved / n,
        "mean_reward": reward_sum / n,
    }


def build_figure(points: list[dict], out_png: str) -> bool:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[ckpt] matplotlib unavailable; skipping figure")
        return False
    steps = [p["step"] for p in points]
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.plot(
        steps,
        [p["structural_valid_rate"] for p in points],
        "o-",
        label="structural: valid-partition rate",
    )
    ax.plot(
        steps,
        [p["semantic_groups_correct"] for p in points],
        "s-",
        label="semantic: correct groups / 4",
    )
    ax.plot(steps, [p["solve_rate"] for p in points], "^--", alpha=0.6, label="solve rate")
    ax.set_xlabel("GRPO training step (step 0 = SFT init)")
    ax.set_ylabel("rate (held-out val split)")
    ax.set_ylim(-0.02, 1.02)
    ax.set_title("GRPO optimizes structure and trades away semantics")
    ax.legend(loc="center right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_png, dpi=180)
    print(f"wrote {out_png}")
    return True


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--arms",
        required=True,
        help="comma list of served_name:step, e.g. 'sft-7b:0,ckpt-50:50,grpo-7b:403'",
    )
    ap.add_argument("--puzzles", required=True)
    ap.add_argument("--out", required=True, help="output prefix (writes .json and .png)")
    ap.add_argument("--max-tokens", type=int, default=512)
    args = ap.parse_args(argv)

    puzzles = load_puzzles(args.puzzles)
    points = []
    for spec in args.arms.split(","):
        name, _, step = spec.strip().rpartition(":")
        metrics = evaluate_arm_components(name, puzzles, args.max_tokens)
        point = {"arm": name, "step": int(step), **metrics}
        points.append(point)
        print(
            f"step {int(step):>4d} ({name}): structural={metrics['structural_valid_rate']:.3f}  "
            f"semantic={metrics['semantic_groups_correct']:.3f}  "
            f"solve={metrics['solve_rate']:.3f}  reward={metrics['mean_reward']:.3f}"
        )
    points.sort(key=lambda p: p["step"])
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out + ".json").write_text(json.dumps(points, indent=1))
    print(f"wrote {args.out}.json")
    build_figure(points, args.out + ".png")


if __name__ == "__main__":
    main()
