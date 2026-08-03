# Results

All arms evaluated on the same held-out test split: **162 puzzles, 2025-12-15 →
2026-05-29**, strictly after every training date. Greedy decoding unless noted.
Bracketed values are bootstrap 95% CIs (1000 resamples).

**Units.** Two conventions appear in this repo and are labeled explicitly in every
table below, because they are easy to confuse:

- `results/` and `results-7b/` (`metrics.json`) report **`groups correct` as a
  mean count on a 0-4 scale**. A value of 0.346 means the model gets 0.346 of the
  4 groups per board on average, i.e. **8.6% of groups**, not 34.6%.
- `results-analysis/` (pass@k, checkpoint curve, entropy/KL) reports the same
  quantity as a **0-1 fraction**, already divided by 4.

Paired differences between arms are quoted on the **0-4 count scale** throughout.

**Measurement sessions.** Some arms were measured twice under different vLLM
serving configurations. Tables state which session they draw from:

- **Session A (main run)**: `results/`, `results-7b/`.
- **Session B (seed-eval run)**: `results-seeds-1.5b/`, `results-seeds-7b/`, and
  the aggregate `results-seeds/seed_summary.json`.

The two sessions disagree slightly on the SFT arms and not at all on the GRPO
arms. See [Reproducibility and measurement noise](#reproducibility-and-measurement-noise)
for the full delta. A statistic is never mixed across sessions within a single
comparison.

## Qwen2.5-1.5B

Session A (`results/`). `Groups correct` is a mean count on a 0-4 scale; the
percentage of groups actually solved is that value divided by 4.

| Arm | n | Solve rate | Groups correct (0-4) | % of groups | Invalid rate | Mean reward |
|---|---|---|---|---|---|---|
| base | 162 | 0.0% [0.0, 0.0] | 0.006 | 0.15% | 32.1% [24.7, 38.9] | 0.049 |
| SFT (LoRA) | 162 | 0.0% [0.0, 0.0] | 0.012 | 0.31% | 74.1% [67.3, 80.2] | −0.038 |
| GRPO (seed 0) | 162 | 0.0% [0.0, 0.0] | 0.006 | 0.15% | **2.5% [0.6, 5.6]** | **0.113** |

Paired per-puzzle reward: GRPO − SFT = +0.152 [0.133, 0.169]; GRPO − base =
+0.064 [0.046, 0.082]. McNemar on solve rate is degenerate (no arm solves
anything).

## Qwen2.5-7B

Session A (`results-7b/`). `Groups correct` is a mean count on a 0-4 scale.

| Arm | n | Solve rate | Groups correct (0-4) | % of groups | Invalid rate | Mean reward |
|---|---|---|---|---|---|---|
| base | 162 | 0.0% | 0.160 | 4.0% | 6.8% [3.1, 11.1] | 0.165 |
| SFT (QLoRA) | 162 | **1.2%** (2/162) | **0.346** | **8.6%** | 22.2% [16.0, 28.4] | **0.197** |
| GRPO (seed 0) | 162 | 0.0% | 0.025 | 0.6% | **0.6% [0.0, 1.9]** | 0.125 |

Paired SFT − GRPO on groups correct (0-4 scale), both arms from session A:
**+0.321 [0.216, 0.432]**.

The corresponding session B figure is +0.296 [0.191, 0.407], and it is the one
quoted in the seed table below, where the SFT baseline is also session B. The two
are the same comparison measured in two serving configurations, not two different
comparisons.

## Reference points (from gvc-local, 8B, multi-agent prompting)

| Arm | n | Solve rate |
|---|---|---|
| gvc-local basic (8B) | 10 | 20.0% [0.0, 50.0] |
| gvc-local GVC multi-agent (8B) | 10 | 60.0% [30.0, 90.0] |

Small n; these are context for the capability ceiling, not matched comparisons.

## Seed replication (3 GRPO seeds per scale)

Each seed re-runs GRPO from the same SFT warm start, isolating RL run-to-run
variance. Same test split, greedy. **Session B** throughout
(`results-seeds-{7b,1.5b}/`, aggregated in `results-seeds/seed_summary.json`).
`Groups correct` is a mean count on a 0-4 scale.

| Scale | Metric | seed 0 | seed 1 | seed 2 | mean ± sd |
|---|---|---|---|---|---|
| 7B | groups correct | 0.025 | 0.043 | 0.068 | 0.045 ± 0.022 |
| 7B | invalid rate | 0.006 | 0.019 | 0.012 | 0.012 ± 0.006 |
| 7B | mean reward | 0.125 | 0.129 | 0.141 | 0.132 ± 0.008 |
| 1.5B | groups correct | 0.006 | 0.000 | 0.000 | 0.002 ± 0.004 |
| 1.5B | invalid rate | 0.025 | 0.031 | 0.037 | 0.031 ± 0.006 |
| 1.5B | mean reward | 0.113 | 0.110 | 0.109 | 0.111 ± 0.002 |

Paired SFT − GRPO on groups correct (0-4 scale), 7B, per seed: +0.296 [0.191,
0.407], +0.278 [0.173, 0.389], +0.253 [0.142, 0.370]. These use the **session B**
SFT baseline (`results-seeds-7b/sft/`, groups correct 0.321), which is the only
one measured alongside seeds 1 and 2, so the three comparisons are internally
consistent. All three 7B seeds fall below base on both grouping (max 0.068 vs
0.160) and reward (max 0.141 vs 0.165).

## pass@16 (temperature 0.9, best-of-k scoring)

Distinguishes capability loss from greedy-decoding degradation.

Unlike the tables above, `results-analysis/` reports grouping as a **0-1
fraction**, so these values are directly percentages of groups.

| Scale | Arm | pass@16 solve | best-of-16 groups correct (fraction) | any-valid |
|---|---|---|---|---|
| 7B | base | 0.000 [0.000, 0.000] | 0.114 | 0.994 |
| 7B | SFT | **0.043** [0.012, 0.074] | **0.252** | 1.000 |
| 7B | GRPO | 0.012 [0.000, 0.031] | 0.022 | 1.000 |
| 1.5B | base | 0.000 | 0.011 | 0.901 |
| 1.5B | SFT | 0.000 | 0.014 | 0.617 |
| 1.5B | GRPO | 0.000 | 0.002 | 0.988 |

Paired bootstrap on best-of-16 groups correct (7B), converted to the **0-4 count
scale** for comparability with the tables above: SFT − GRPO = +0.920 [+0.772,
+1.062]; base − GRPO = +0.370 [+0.259, +0.475]. On the fraction scale those are
+0.230 and +0.093. Sampling does not recover the
capability, so the loss is distributional rather than a decoding artifact.

*Only k=16 has been measured. The full k ∈ {1, 2, 4, 8, 16, 32} curve is pending.*

## Checkpoint decomposition (7B, val split, greedy)

Reward split into its structural and semantic components over GRPO training.

| Step | Structural (valid partition) | Semantic (groups correct / 4) | Solve | Mean reward |
|---|---|---|---|---|
| 0 (SFT init) | 0.787 | 0.102 | 0.000 | 0.207 |
| 50 | 0.926 | 0.139 | 0.000 | **0.289** |
| 100 | 0.861 | 0.120 | 0.009 | 0.256 |
| 150 | 0.972 | **0.012** | 0.000 | 0.129 |
| 200-403 | 0.963-0.972 | 0.009-0.012 | 0.000 | 0.123-0.126 |

Both components improve through step 100; between 100 and 150 semantics collapses
an order of magnitude while structure locks in and never moves again. Mean reward
peaks at step 50 and ends *below* that peak, so the final policy is not even
reward-optimal on held-out data.

Evaluated on the **val** split so the test set is never reused for analysis.
1.5B checkpoints were not Hub-synced during the original run and are
unrecoverable, so this curve is 7B-only.

## Policy entropy and KL from reference (7B, val split, temperature 0.9, n=100)

Entropy is the mean per-token entropy of the policy's own distribution on its own
samples. KL is the sample-based estimate E_{y~pi}[log pi(y|x) − log ref(y|x)].
Sampled rather than greedy, so absolute structural and semantic values are not
comparable to the greedy tables above; the trajectory is what matters.

| Step | Entropy (nats/tok) | KL from SFT (nats/seq) | KL from base (nats/tok) | Structural | Semantic |
|---|---|---|---|---|---|
| base | 0.258 | 42.60 | 0.000 | 0.69 | 0.013 |
| 0 (SFT init) | 0.303 | 0.00 | 0.395 | 0.37 | 0.033 |
| 50 | 0.211 | 2.70 | 0.342 | 0.68 | **0.095** |
| 100 | 0.200 | 8.12 | 0.336 | 0.64 | 0.088 |
| 150 | 0.016 | 46.42 | 0.139 | 0.95 | 0.008 |
| 200 | 0.011 | 46.88 | 0.133 | 0.95 | 0.010 |
| 250 | 0.010 | 46.96 | 0.130 | 0.96 | 0.010 |
| 300-403 | 0.0099 | 47.05 | 0.130 | 0.96 | 0.010 |

Three readings. **Entropy collapses 12.7x in the single step 100 to 150 interval**
(30.6x over the whole run from the SFT init), the same interval in which the
greedy decomposition above loses semantics. **KL saturates**: 98.7% of the total
displacement from the SFT init is spent by step 150, and the remaining 253 steps
add 1.4%; generated token counts are identical (7928) from step 200 onward.
**The over-optimization curve is an inverted U**: semantic score peaks at 0.095 at
KL 2.70 nats/sequence and falls 9.5x to 0.010 by KL 47.

Converting to the reference's cross-entropy on the policy's samples (CE = KL + H):
the untrained base assigns 0.140 nats/token to the final GRPO output versus 0.698
to SFT's, so the collapsed text is 5.0x more predictable to the base model than
SFT's is, and 1.85x more predictable than the base model's own samples (0.258).

Seed 0 checkpoints only (the only seed Hub-synced during training).

## Weight-space seed convergence

Cosine similarity between independent seeds' RL-induced effective LoRA updates
(dW = B_grpo·A_grpo − B_sft·A_sft):

| | 1.5B | 7B |
|---|---|---|
| update magnitude, 3 seeds | 0.306 / 0.304 / 0.303 (0.13× SFT) | 0.273 / 0.267 / 0.277 (0.09× SFT) |
| cos(seed0, seed1) | +0.779 | +0.679 |
| cos(seed0, seed2) | +0.797 | +0.675 |
| cos(seed1, seed2) | +0.793 | +0.688 |
| cos(dW_RL, dW_SFT) control | −0.026 / −0.029 / −0.028 | +0.002 / −0.007 / −0.001 |
| random-direction baseline | 2.2e−5 | 9.9e−6 |
| per-module cos, median | +0.771 (min +0.187) | +0.644 (min −0.039) |

Largest per-module changes concentrate in mid-layer MLP `up_proj`/`gate_proj`
(layers 9-18) at both scales, with per-module cosines of 0.80-0.91.

## Reproducibility and measurement noise

Two distinct observations, both worth stating.

**Identical serving config reproduces exactly.** The published 1.5B eval was
re-run end-to-end on a fresh VM with the same vLLM configuration and reproduced
byte-identically under greedy decoding.

**Different serving config introduces small drift — but only for high-entropy
policies.** The seed-replication session (B) served four LoRA adapters at once
(different `--max-loras`, different tensor-parallel layout) and re-measured the
SFT arms as a side effect. Comparing the same adapter across the two sessions on
**every** metric, not just grouping:

| Arm | Metric | Session A | Session B | Δ |
|---|---|---|---|---|
| 7B SFT | solve rate | 0.012346 | 0.012346 | **0.000000** |
| 7B SFT | groups correct | 0.345679 | 0.320988 | 0.024691 |
| 7B SFT | invalid rate | 0.222222 | 0.228395 | −0.006173 |
| 7B SFT | mean reward | 0.196914 | 0.188580 | 0.008333 |
| 1.5B SFT | groups correct | 0.012346 | 0.018519 | −0.006173 |
| 1.5B SFT | mean reward | −0.038272 | −0.036728 | −0.001543 |
| 1.5B SFT | invalid rate | 0.740741 | 0.740741 | **0.000000** |
| 7B GRPO seed 0 | all four | — | — | **0.000000** |
| 1.5B GRPO seed 0 | all four | — | — | **0.000000** |

Greedy decoding is not bitwise-deterministic across vLLM batching/parallelism
configurations, so a few borderline tokens flip. At the per-puzzle level the
7B SFT arm differs on **2 of 162 puzzles** for grouping and **3 of 162** for
format validity; the GRPO arms differ on **0 of 162** for every metric. Practical
measurement noise on `groups_correct` is therefore ~0.025 (0-4 scale) for SFT
arms, an order of magnitude smaller than the effects claimed here
(SFT − GRPO ≈ 0.25-0.32 at 7B), so no conclusion is affected.

This is why the headline 7B table reads 0.346 while `seed_summary.json` reads
0.321 for what is nominally the same arm. Neither file is stale. They are two
serving configurations of the same adapter on the same 162 puzzles, and the solve
rate (2/162) is identical in both because the two puzzles that arm solves are not
close calls.

Note which arms drift: the **GRPO adapters reproduce exactly** while the SFT
adapters do not. That is an independent corroboration of the entropy-collapse
story: the final 7B GRPO policy's measured entropy of 0.0099 nats/token leaves no
borderline token decisions to flip, whereas the SFT policy has the highest entropy
of any arm (0.303).

Note which arms drift: the **GRPO adapters reproduce exactly** while the SFT
adapters do not. That is an independent corroboration of the entropy-collapse
story — a policy with entropy ~3e−4 emits the same tokens regardless of serving
nondeterminism, whereas a higher-entropy policy has borderline decisions to flip.
Where tables above report SFT numbers, they use the main-run measurement.

## Provenance

| Table | Source |
|---|---|
| 1.5B, 7B main results | `results/`, `results-7b/` |
| Seed replication | `results-seeds-7b/`, `results-seeds-1.5b/`, `results-seeds/seed_summary.json` |
| pass@16 | `results-analysis/passk-7b.json`, `passk-1.5b.json` |
| Checkpoint decomposition | `results-analysis/ckpt-curve-7b.json` + `.png` |
| Entropy + KL per checkpoint | `results-analysis/entropy-kl-7b.json` + `.png` |
| Weight-space convergence | `results-seeds/weight_space_{7b,1.5b}.txt` |
| Training configuration | `report/implementation_notes.md` |
