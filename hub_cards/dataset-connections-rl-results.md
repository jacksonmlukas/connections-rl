---
license: mit
language:
  - en
pretty_name: connections-rl evaluation artifacts
size_categories:
  - n<1K
task_categories:
  - text-generation
tags:
  - nyt-connections
  - grpo
  - rlvr
  - reward-over-optimization
  - evaluation
  - negative-results
viewer: false
---

# connections-rl: raw evaluation artifacts

Per-puzzle records, bootstrap summaries and analysis outputs backing
[**connections-rl**](https://github.com/jacksonmlukas/connections-rl), a two-scale (Qwen2.5-1.5B / 7B), three-seed study of
what verifiable-reward RL actually transfers.

This is an artifact bundle for auditing published numbers, not a loadable
training dataset, so the dataset viewer is disabled.

## Read this before using the numbers

Two conventions in these files are easy to misread. Both have bitten this
project already.

**1. `groups_correct` has two different scales depending on the directory.**

| Location | Scale | Example |
|---|---|---|
| `results-7b/`, `results-seeds-*/` | **mean count, 0-4** | `0.346` means 8.6% of groups |
| `results-analysis/passk-*.json` | **fraction, 0-1** (already divided by 4) | `0.252` means 25.2% of groups |
| `results-analysis/ckpt-curve-*.json`, `entropy-kl-*.json` | **fraction, 0-1** | field is named `semantic_groups_correct` |

Each board has 4 groups. A 0-4 count of 0.346 is
**8.6%** of groups, not 34.6%.
Paired differences quoted in the write-up use the 0-4 count scale.

**2. Every metric under `summary` is `[point_estimate, ci_lower, ci_upper]`, not three seeds.**

A three-element array looks like per-seed values. It is a percentile bootstrap
over 1,000 resamples (`stats.bootstrap_ci`, `alpha=0.05`, `seed=0`), returning
`(mean, lower, upper)`. Worked example from `results-7b/base/metrics.json`:

```json
"groups_correct": [0.16049, 0.10494, 0.22222]
```

means the untrained 7B base solves **0.160 of 4 groups per board**
(4.0% of groups), 95% CI **[0.105, 0.222]** on the same 0-4
scale. Per-seed values live in `results-seeds/seed_summary.json`, which uses
plain scalars under `per_arm`, and explicit `mean`/`sd` under `across_seed`.

## Repository layout

```
results-7b/                  Qwen2.5-7B main eval (session A), n=162 test puzzles
  base|sft|grpo/
    metrics.json             bootstrap summary, stratified + OVERALL
    records.jsonl            one row per puzzle
  comparisons.json           McNemar + paired-bootstrap between arms
results-seeds-7b/            7B seed replication (session B)
  sft|grpo-seed0|1|2/        metrics.json + records.jsonl
results-seeds-1.5b/          1.5B seed replication (session B)
  sft|grpo-seed0|1|2/        metrics.json + records.jsonl
results-seeds/
  seed_summary.json          per-arm scalars + across-seed mean/sd
  weight_space_7b.txt        cross-seed LoRA update cosine similarity
  weight_space_1.5b.txt
results-analysis/
  passk-7b.json              pass@16, temperature 0.9, best-of-k scoring
  passk-1.5b.json
  ckpt-curve-7b.json/.png    structure vs semantics over GRPO training (val)
  entropy-kl-7b.json/.png    policy entropy + KL from SFT init and base (val)
```

**Not in this repo:** `results/`, the 1.5B main-run eval (session A). It lives in
the GitHub repository under [`results/`](https://github.com/jacksonmlukas/connections-rl/tree/main/results). The 1.5B
numbers quoted in the write-up come from there; the 1.5B files here are the
seed-replication session.

## File schemas

`metrics.json`

| Field | Meaning |
|---|---|
| `arm` | arm name |
| `n` | puzzles evaluated (162 for all test-split runs) |
| `n_resamples` | bootstrap resamples (1000) |
| `summary.OVERALL.<metric>` | `[point, ci_lo, ci_hi]` for `solve_rate`, `groups_correct`, `one_away_rate`, `invalid_rate`, `reward` |
| `summary.<stratum>.<metric>` | same, per puzzle category (`wordplay`, `cultural`, `category`, `tag-fillin`, `silent-letter`) |
| `groups_correct_distribution` | histogram of exact groups solved, keys `"0"`-`"4"` |

`records.jsonl`, one JSON object per puzzle:

| Field | Type | Meaning |
|---|---|---|
| `puzzle_id`, `date`, `strata` | int, ISO date, str | puzzle identity and category |
| `solved` | bool | all 4 groups correct |
| `groups_correct` | int 0-4 | **count**, not a fraction |
| `one_away` | bool | exactly one group off |
| `invalid_format` | bool | malformed output or words not on the board |
| `reward` | float | deterministic reward, max 1.6 |
| `latency_ms` | float | generation wall time |

`comparisons.json`: `mcnemar_p` (exact, on solve/no-solve) and
`solve_rate_diff_ci` as `[diff, lo, hi]` for `a - b`, plus discordant-pair counts.

`passk-*.json`: `arms.<arm>.summary` holds `pass_at_k_solve`,
`pass_at_k_valid`, `best_of_k_groups_correct`, each `[point, lo, hi]`;
`arms.<arm>.records` holds per-puzzle `max_groups_correct` (an int 0-4) and
`any_solved` / `any_valid`.

## Measurement sessions

Some arms were measured twice under different vLLM serving configurations:

- **Session A**: `results-7b/` (and `results/` on GitHub).
- **Session B**: `results-seeds-7b/`, `results-seeds-1.5b/`, `results-seeds/`.

The 7B SFT arm reads 0.346 in session A and
0.321 in session B for the same adapter.
Neither is stale. Greedy decoding is not bitwise deterministic across vLLM
batching and parallelism layouts, so the two sessions differ on 2 of 162 puzzles
for grouping and 3 of 162 for validity. **Every GRPO arm reproduces exactly**
(0 of 162 on all metrics), which corroborates the entropy-collapse finding: the
final 7B GRPO policy sits at 0.0099 nats/token
and has no borderline decisions to flip. Do not mix sessions inside one
comparison.

## Headline numbers these files support

Held-out test split, 162 puzzles, 2025-12-15 to 2026-05-29, strictly after every
training date. Greedy decoding.

| Arm (7B) | Solve rate | Groups correct (0-4) | % of groups | Invalid | Mean reward |
|---|---|---|---|---|---|
| base | 0.0% | 0.160 | 4.0% | 6.8% | 0.165 |
| SFT | 1.2% | 0.346 | 8.6% | 22.2% | 0.197 |
| GRPO | 0.0% | 0.025 | 0.6% | 0.6% | 0.125 |

GRPO reaches the best structural validity of any arm while collapsing grouping
below the untrained base. Full analysis in
[`report/findings.md`](https://github.com/jacksonmlukas/connections-rl/blob/main/report/findings.md).

## Usage

```python
import json
from huggingface_hub import hf_hub_download, snapshot_download

p = hf_hub_download("jacksonlukas/connections-rl-results",
                    "results-7b/base/metrics.json", repo_type="dataset")
m = json.load(open(p))["summary"]["OVERALL"]

point, lo, hi = m["groups_correct"]          # [point, ci_lo, ci_hi], 0-4 scale
print(f"{point:.3f} of 4 groups = {100 * point / 4:.1f}% of groups, 95% CI [{lo:.3f}, {hi:.3f}]")

# per-puzzle records
local = snapshot_download("jacksonlukas/connections-rl-results", repo_type="dataset")
rows = [json.loads(l) for l in open(f"{local}/results-7b/base/records.jsonl")]
print(sum(r["groups_correct"] for r in rows) / len(rows))   # reproduces `point`
```

## Provenance and licensing

Generated by the evaluation harness in
[`src/connections_rl/eval`](https://github.com/jacksonmlukas/connections-rl/tree/main/src/connections_rl/eval); the same
files are committed in the GitHub repository, which is the source of truth.
Derived from the NYT Connections puzzle database in
[gvc-local](https://github.com/jacksonmlukas/gvc-local). These are model outputs
and aggregate statistics, not puzzle content redistribution. Released under MIT;
NYT Connections puzzles remain the property of The New York Times.

## Citation

If you use this adapter or the accompanying analysis, please cite the repository:

```bibtex
@software{lukas_connections_rl_2026,
  author = {Lukas, Jackson},
  title  = {connections-rl: What Verifiable-Reward RL Actually Transfers},
  year   = {2026},
  url    = {https://github.com/jacksonmlukas/connections-rl},
  note   = {Two-scale, three-seed GRPO study on NYT Connections}
}
```

The predecessor multi-agent work is published as
[Snap Out of It (ACL 2025, REALM Workshop)](https://aclanthology.org/2025.realm-1.16/).

## Contact

Open an issue at [https://github.com/jacksonmlukas/connections-rl/issues](https://github.com/jacksonmlukas/connections-rl/issues).
