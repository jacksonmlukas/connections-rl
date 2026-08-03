# connections-rl

**GRPO post-training a small open model on NYT Connections, measured with leakage-aware evaluation.**

**Key finding: verifiable-reward RL transfers exactly what the reward can verify — and whether that helps depends on model scale.** At 1.5B, GRPO cut invalid outputs **74.1% → 2.5%** on held-out puzzles (pure gain: the model had no grouping ability to lose). At 7B, the *same* training flipped net-harmful: best format validity of any arm (0.6% invalid) but grouping collapsed below the untrained base, while SFT alone delivered the first held-out solves. In both cases training reward saturated at its theoretical maximum with policy entropy → 0 — memorization of the 807 training answers. **Replicated over 3 GRPO seeds per scale**, with the direction of the effect identical in all six runs. A measured, two-scale case study in reward over-optimization: bootstrap CIs, McNemar paired tests, per-seed variance, and a strictly chronological held-out test set.

![Invalid-output rate by arm](invalid_rate.png)

**How much of a multi-agent system's gain can a single small open model recover with RL post-training?**

My [ACL 2025 paper](https://aclanthology.org/2025.realm-1.16/) (REALM Workshop; equal-contribution co-author) showed a multi-agent GPT-4o loop solves NYT Connections at 98%, and [gvc-local](https://github.com/jacksonmlukas/gvc-local) pushed an open 8B model to 60% with multi-agent prompting. This repo answers the follow-up: post-train small open models (**Qwen2.5-1.5B and 7B**) directly with GRPO (verifiable-reward RL, DeepSeek-R1 style) and measure them against those baselines with a production-grade evaluation stack.

## Results

Held-out test set: 162 puzzles, strictly *after* every training date (2025-12-15 → 2026-05-29).

| Arm | n | Solve rate (95% CI) | Invalid rate (95% CI) | Mean reward |
| --- | --- | --- | --- | --- |
| gvc-local basic (8B, reference) | 10 | 20.0% [0.0, 50.0] | — | — |
| gvc-local GVC multi-agent (8B, reference) | 10 | 60.0% [30.0, 90.0] | — | — |
| base (Qwen2.5-1.5B) | 162 | 0.0% [0.0, 0.0] | 32.1% [24.7, 38.9] | 0.049 |
| SFT (LoRA) | 162 | 0.0% [0.0, 0.0] | 74.1% [67.3, 80.2] | −0.038 |
| **GRPO** (seed 0) | 162 | 0.0% [0.0, 0.0] | **2.5% [0.6, 5.6]** | **0.113** |

**Headline finding (honest negative result):** at 1.5B, no arm solves any held-out puzzle. GRPO transfers exactly what the verifiable reward can verify. Training reward saturated at the theoretical maximum (1.6: perfect format + all four groups + solve bonus) with policy entropy collapsing to ~0, i.e. the model *memorized* the 807 training answers. On unseen boards the grouping ability doesn't transfer, but the format/board-grounding discipline does: invalid outputs (hallucinated words, malformed answers) drop from 74.1% (SFT) and 32.1% (base) to **2.5%**, and the paired per-puzzle reward gain is significant (+0.152 vs SFT, 95% CI [0.133, 0.169]; +0.064 vs base, [0.046, 0.082]). Full narrative in [`report/`](https://github.com/jacksonmlukas/connections-rl/blob/main/report).

### Scale ablation: Qwen2.5-7B (same data, same reward, same protocol)

| Arm | n | Solve rate | Groups correct (mean) | Invalid rate (95% CI) | Mean reward |
|---|---|---|---|---|---|
| base (Qwen2.5-7B) | 162 | 0.0% | 0.160 | 6.8% [3.1, 11.1] | 0.165 |
| SFT (QLoRA) | 162 | **1.2%** (2/162) | **0.346** | 22.2% [16.0, 28.4] | **0.197** |
| GRPO (seed 0) | 162 | 0.0% | 0.025 | **0.6% [0.0, 1.9]** | 0.125 |

Scale unlocks real competence (base gets 16% of groups; SFT doubles it and produces the first held-out solves), but **GRPO flips from net-positive to net-harmful**: it achieves the best format validity of any arm at any scale (0.6% invalid) while collapsing grouping ability *below the untrained base* — mean reward drops under base. Same memorization mechanism as 1.5B; at 7B there was actual semantic ability to trade away. The cross-scale conclusion: GRPO against this reward optimizes exactly what the reward verifies (structure) at the expense of what it can't (semantics), and whether that trade helps or hurts depends on how much semantic ability the starting policy had.

### Seed replication (3 GRPO seeds per scale)

Each seed re-runs GRPO from the same SFT warm start, isolating RL run-to-run variance. Held-out test split, greedy decoding, n=162.

| Scale | Metric | seed 0 | seed 1 | seed 2 | mean ± sd |
|---|---|---|---|---|---|
| 7B | groups correct | 0.025 | 0.043 | 0.068 | 0.045 ± 0.022 |
| 7B | invalid rate | 0.006 | 0.019 | 0.012 | 0.012 ± 0.006 |
| 7B | mean reward | 0.125 | 0.129 | 0.141 | 0.132 ± 0.008 |
| 1.5B | groups correct | 0.006 | 0.000 | 0.000 | 0.002 ± 0.004 |
| 1.5B | invalid rate | 0.025 | 0.031 | 0.037 | 0.031 ± 0.006 |
| 1.5B | mean reward | 0.113 | 0.110 | 0.109 | 0.111 ± 0.002 |

Both headline effects replicate in every run. At 7B, all three seeds fall below base on grouping (max 0.068 vs base 0.160) and below base on reward (max 0.141 vs 0.165); paired bootstrap of SFT − GRPO on groups correct gives +0.296 [0.191, 0.407], +0.278 [0.173, 0.389], +0.253 [0.142, 0.370] — three independent CIs excluding zero. At 1.5B, all three seeds hold invalid rate near 3% (vs SFT 74.1%) with reward above base. Seed 0, the originally published run, is the *least* favorable 7B draw on grouping, so the headline table understates GRPO rather than cherry-picking.

**Weight-space convergence.** Independent seeds do not merely agree behaviorally — they move the policy in the same direction. Cosine similarity between seeds' RL-induced LoRA updates is **+0.68 to +0.69 at 7B** and **+0.78 to +0.80 at 1.5B**, against a random-direction expectation of ~1e−5 and a near-zero control against the SFT update direction (so this is not an artifact of the shared warm start). Update magnitudes agree within 4%, and the largest changes concentrate in the same mid-layer MLP `up_proj`/`gate_proj` modules at both scales. The collapse is a systematic attractor of this reward under this optimizer, not seed noise. See [`results-seeds/`](https://github.com/jacksonmlukas/connections-rl/tree/main/results-seeds).

All arms are evaluated on the same **leakage-aware, date-split held-out test set** with bootstrap CIs, McNemar significance tests between arms, and per-stratum breakdowns. CI re-runs the eval smoke and a release gate (GRPO must not regress vs. SFT beyond the CI) on every push.

## How it works

1. **Data** — reuses gvc-local's tagged puzzle DB (1,078 puzzles, 2023-06 → 2026-05). Splits are strictly chronological: everything trained on predates everything tested on.
2. **Reward** (`connections_rl/reward`) — deterministic and unit-tested: format validity (all 16 board words, 4×4, once each), fully-correct groups / 4, a solve bonus, optional one-away shaping, and a penalty for malformed output.
3. **SFT warm start** (`make train-sft`) — rank-16 LoRA on the train split.
4. **GRPO** (`make train-grpo`) — K=8 completions per puzzle, group-relative advantage, KL penalty to the SFT reference. Single-GPU on a free Colab/Kaggle T4; QLoRA for 7B. Checkpoints sync to the Hub so runs resume across ephemeral sessions. Seed replicates via `--seed/--output-dir/--ckpt-hub-repo` overrides (`notebooks/kaggle_seed_run.ipynb`). Exact normalization settings are recorded in [`report/implementation_notes.md`](report/implementation_notes.md).
5. **Eval** (`make eval`) — stratified sampling, bootstrap CIs, paired significance tests, reliability/ECE utilities. Beyond the main table: `eval/passk.py` (best-of-k sampling), `eval/checkpoint_curve.py` (structure/semantics decomposition over training), and `scripts/analyze_seed_weightspace.py` (cross-seed weight-space convergence). All results committed under `results*/`.
6. **Serving** (`make serve`) — FastAPI over vLLM with `/solve`, `/compare` (base vs. GRPO on the same board), `/health`, `/metrics`. `docker compose up` runs the full stack.

## Quickstart

```
git clone https://github.com/jacksonmlukas/connections-rl && cd connections-rl
make setup                 # pip install -e ".[dev]"
export CONNECTIONS_PUZZLES=path/to/gvc-local/data/puzzles/tagged_connections.json
make data                  # leakage-aware splits + SFT chat data
make test lint             # unit tests + ruff + mypy
make eval-smoke            # end-to-end harness check, no GPU needed
```

Training (GPU): open `notebooks/colab_grpo.ipynb` on Colab, or:

```
pip install -e ".[train]"
make train-sft && make train-grpo
```

Serving:

```
docker compose up          # vLLM + API
curl -X POST localhost:8080/compare -H 'content-type: application/json' \
  -d '{"words": ["HAIL","RAIN","SLEET","SNOW","BUCKS","HEAT","JAZZ","NETS","OPTION","RETURN","SHIFT","TAB","KAYAK","LEVEL","MOM","RACECAR"]}'
```

## Repo map

```
configs/        model / train / eval / accelerate configs (incl. 7B + per-seed eval)
src/connections_rl/
  data/         puzzle loading, date splits, chat formatting
  reward/       verifiable reward (the RL core)
  train/        sft.py (LoRA/QLoRA), grpo.py (TRL GRPOTrainer + seed overrides)
  eval/         harness, bootstrap/McNemar/ECE stats, release gate,
                passk.py, checkpoint_curve.py
  serve/        FastAPI + vLLM serving, request monitoring
  report/       results table + plots
scripts/        weight-space seed analysis, model-card push
notebooks/      Colab/Kaggle runbooks: training, 7B, analysis, seed runs, seed eval
hub_cards/      model cards for the 8 published adapters
results/        1.5B main results          results-7b/     7B main results
results-seeds*/ per-seed metrics + weight-space convergence
results-analysis/ pass@k + checkpoint decomposition
report/         technical writeup, results tables, implementation notes
```

## Related

- [gvc-local](https://github.com/jacksonmlukas/gvc-local) — multi-agent prompting predecessor; source of the puzzle DB and the 60% baseline.
- [Snap Out of It (ACL 2025, REALM Workshop)](https://aclanthology.org/2025.realm-1.16/) — multi-agent GPT-4o loop at 98%; equal-contribution co-author.

MIT license.
