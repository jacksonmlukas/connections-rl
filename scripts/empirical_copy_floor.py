#!/usr/bin/env python3
"""Empirical copier ceiling: how many consecutive quadruples of the SERVED
(puzzle-seeded shuffled) presentations are true answer groups, per split.

The 1.4-slot chance floor (tex, S3) is the expectation under a uniformly
drawn valid partition. The presentations actually served are one fixed draw
of the puzzle-seeded shuffle, so a positional copier's score is bounded by
the REALIZED alignment count, not the expectation. This script enumerates it
exactly -- no GPU, no sampling.

Run from the repo root after `make data`:
  python3 scripts/empirical_copy_floor.py
Writes results-analysis/sep06/empirical_floor.json.
"""
import json

from connections_rl.data.formatting import shuffled_words
from connections_rl.data.loader import load_puzzles

out = {"note": "aligned = consecutive quadruple (positions 1-4/5-8/9-12/13-16) "
               "of the served shuffled presentation equals a true group as a set; "
               "uniform_expectation = boards * 16/1820 (= 4 positions x 4/C(16,4))"}
for split in ["train", "val", "test"]:
    puzzles = load_puzzles(f"data/splits/puzzles_{split}.json")
    total, boards = 0, []
    for p in puzzles:
        w = shuffled_words(p)
        keys = [frozenset(g.members) for g in p.groups]
        c = sum(1 for i in range(0, 16, 4) if frozenset(w[i : i + 4]) in keys)
        total += c
        if c:
            boards.append({"puzzle_id": p.puzzle_id, "aligned_quadruples": c})
    out[split] = {
        "n_boards": len(puzzles),
        "aligned_quadruples_realized": total,
        "uniform_expectation": round(len(puzzles) * 16 / 1820, 3),
        "aligned_boards": boards,
    }
    print(split, out[split]["n_boards"], "boards -> realized", total,
          "vs uniform expectation", out[split]["uniform_expectation"])

json.dump(out, open("results-analysis/sep06/empirical_floor.json", "w"), indent=1)
print("wrote results-analysis/sep06/empirical_floor.json")
