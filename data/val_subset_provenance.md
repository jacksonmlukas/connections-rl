# R5 — which 100 of the 108 validation puzzles the n=100 curve uses

**Mechanism: a head-slice, not a sample.** The n=100 series comes from
`src/connections_rl/eval/entropy_kl.py`, whose `--n` flag (default 100;
`entropy_kl.py:257`) is applied as a plain prefix slice:

```python
puzzles = load_puzzles(args.puzzles)[: args.n]   # entropy_kl.py:292
```

`load_puzzles` sorts by `(date, puzzle_id)` before returning
(`src/connections_rl/data/loader.py:110`), so the slice keeps the 100
**chronologically earliest** validation puzzles and drops the 8 most recent.
The `--n` help string says "puzzles to sample," but nothing is sampled — no
RNG touches puzzle selection.

**Which 100.** The validation split is puzzle ids 809–916 (2025-08-28 to
2025-12-14; contiguous, re-derived 2026-08-21 from
`gvc-local/data/puzzles/raw_connections.json` through the repo's own
`split_by_date` logic, 807/108/162 with test = ids 917–1078 matching every
published table). The curve therefore uses **ids 809–908** (2025-08-28 through
2025-12-06) and excludes **ids 909–916** (2025-12-07 through 2025-12-14).

**Deterministic across runs: yes**, for puzzle selection — sort plus slice,
no randomness. Generation for the scores is temperature-0.9 sampling, seeded
per puzzle as `torch.manual_seed(seed + p.puzzle_id)` with `--seed` default 0
(`entropy_kl.py:122`, `:260`), i.e. reproducible given the same
software/hardware stack, though not bit-guaranteed across GPU stacks.

**One correction to the brief's pointer.** The n=100 mechanism is *not* in
`eval/checkpoint_curve.py`: that module evaluates the full file it is given
(no `--n` flag) and `results-analysis/ckpt-curve-7b.json` is n=108, greedy.
The published integer-consistent series (base 5, SFT 13, step-50 38, step-100
35, step-150 3, steps 200–403 4, all out of 400) is
`results-analysis/entropy-kl-7b.json` — n_puzzles=100 in every row,
temperature 0.9 — produced by `entropy_kl.py`. The appendix line should cite
the entropy/KL sweep, not the checkpoint curve, as the n=100 surface; the two
curves must not be conflated (different n, different decoding).
