#!/usr/bin/env python3
"""Score an arm on ANSWER-ORDERED held-out prompts (the training presentation).

The paper's held-out detector worked because evaluation prompts were shuffled
while the leaked GRPO training prompts were answer-ordered (tex, S2/S7). That
means the detector succeeded partly by accident: had evaluation shared the
training presentation, the endpoint would have read as a near-perfect solver.
This script measures that counterfactual directly, on the 162-puzzle test
split: it rebuilds each user turn with the sixteen words in answer-key order
-- exactly the presentation the buggy ``build_dataset`` produced -- and runs
the standard harness (same reward, same parser, greedy) against a vLLM
endpoint.

Prespecified prediction (written before the run): if the copy-rule diagnosis
is right, the step-403 endpoint reads ~1.6 mean reward and ~4.0 groups
correct here, against 0.125 / 0.0247 on shuffled prompts (control session,
paper Table 2). A large gap between the two presentations of the SAME
held-out puzzles is the copy rule read as a single number.

Usage (endpoint served as in the paper's eval sessions):
  vllm serve Qwen/Qwen2.5-7B-Instruct --enable-lora \
      --lora-modules connections-rl-grpo-7b=adapters/grpo-7b ...
  python3 scripts/eval_answer_ordered.py \
      --model connections-rl-grpo-7b --arm grpo-final
Cost: one pass over 162 puzzles, greedy -- about one GPU-hour on a T4 pair.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import connections_rl.eval.harness as harness
from connections_rl.data.formatting import SYSTEM_PROMPT
from connections_rl.data.loader import Puzzle, load_puzzles
from connections_rl.eval.solvers import endpoint_solver


def answer_ordered_words(puzzle: Puzzle) -> list[str]:
    """The board in answer-key order: positions 1-4 the first group, and so on.

    Mirrors the bug in the pre-fix ``connections_rl.train.grpo.build_dataset``
    (words taken group by group from the answer key, no shuffle). Group order
    within the key is irrelevant to the copy rule: any consecutive quadruple
    of this presentation is a correct group.
    """
    words = [w for g in puzzle.groups for w in g.members]
    if sorted(words) != sorted(puzzle.words):
        raise AssertionError(
            f"puzzle {puzzle.puzzle_id}: answer-key words != board words"
        )
    return words


def answer_ordered_chat(puzzle: Puzzle, seed: int | None = None) -> list[dict[str, str]]:
    # Same system prompt, same template as data.formatting.build_chat; only
    # the word order differs. ``seed`` is accepted (and ignored) so the
    # harness can call this in build_chat's place.
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": "Words: " + ", ".join(answer_ordered_words(puzzle))},
    ]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--puzzles", default="data/splits/puzzles_test.json")
    ap.add_argument("--model", default="connections-rl-grpo-7b")
    ap.add_argument("--arm", default="grpo-final")
    ap.add_argument("--base-url", default=None)
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--out", default="results-analysis/answer-ordered")
    ap.add_argument("--n-resamples", type=int, default=1000)
    args = ap.parse_args()

    puzzles = load_puzzles(args.puzzles)
    print(f"{len(puzzles)} puzzles from {args.puzzles}; presentation: ANSWER-ORDERED")

    # Reuse the standard harness verbatim -- same reward, parser, records,
    # bootstrap -- with only the prompt builder swapped. This is the same
    # scoring path as every shuffled evaluation in the paper.
    harness.build_chat = answer_ordered_chat

    solver = endpoint_solver(
        model=args.model,
        base_url=args.base_url,
        temperature=0.0,
        max_tokens=args.max_tokens,
    )
    out_dir = Path(args.out) / args.arm
    res = harness.evaluate_arm(
        args.arm,
        solver,
        puzzles,
        capture_path=out_dir / "generations.jsonl",
    )
    res.save(out_dir, args.n_resamples)

    s = res.summary(n_resamples=args.n_resamples)["OVERALL"]
    print(json.dumps({m: v for m, v in s.items()}, indent=1))
    reward, groups = s["reward"][0], s["groups_correct"][0]
    print(
        f"\nANSWER-ORDERED test split, arm={args.arm}: "
        f"mean reward {reward:.4f}, groups correct {groups:.4f} (0-4 scale)"
    )
    print(
        "Shuffled-prompt reference for the step-403 endpoint (control session, "
        "paper Table 2): reward 0.1250, groups 0.0247. The prespecified "
        "prediction for the endpoint HERE is ~1.6 / ~4.0."
    )


if __name__ == "__main__":
    main()
