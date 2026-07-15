"""``make data``: build leakage-aware splits + SFT chat data.

Writes to --out (default data/splits/):
  train.jsonl / val.jsonl / test.jsonl   one SFT chat example per puzzle
  puzzles_{train,val,test}.json          raw puzzle records for eval/GRPO
  split_summary.json                     sizes and date ranges
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from connections_rl.data.formatting import build_sft_example
from connections_rl.data.loader import Puzzle, default_puzzles_path, load_puzzles
from connections_rl.data.splits import split_by_date


def puzzle_to_record(p: Puzzle) -> dict:
    return {
        "puzzle_id": p.puzzle_id,
        "date": p.date,
        "category": p.strata,
        "answers": [
            {"level": g.level, "group": g.name, "members": list(g.members)} for g in p.groups
        ],
    }


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--puzzles", default=None, help="path to tagged_connections.json")
    ap.add_argument("--out", default="data/splits")
    ap.add_argument("--val-frac", type=float, default=0.10)
    ap.add_argument("--test-frac", type=float, default=0.15)
    args = ap.parse_args(argv)

    src = Path(args.puzzles) if args.puzzles else default_puzzles_path()
    puzzles = load_puzzles(src)
    split = split_by_date(puzzles, val_frac=args.val_frac, test_frac=args.test_frac)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for name, part in (("train", split.train), ("val", split.val), ("test", split.test)):
        with open(out / f"{name}.jsonl", "w", encoding="utf-8") as f:
            for p in part:
                f.write(json.dumps(build_sft_example(p)) + "\n")
        with open(out / f"puzzles_{name}.json", "w", encoding="utf-8") as f:
            json.dump([puzzle_to_record(p) for p in part], f, indent=1)
    summary = split.summary()
    with open(out / "split_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"source: {src}")
    for k, v in summary.items():
        print(f"{k:>5}: n={v['n']:>4}  {v['start']} .. {v['end']}")


if __name__ == "__main__":
    main()
