# Results

All arms evaluated on the same held-out test split: **162 puzzles, 2025-12-15 →
2026-05-29**, strictly after every training date. Greedy decoding unless noted.
Bracketed values are bootstrap 95% CIs (1000 resamples).

## Qwen2.5-1.5B

| Arm | n | Solve rate | Groups correct | Invalid rate | Mean reward |
|---|---|---|---|---|---|
| base | 162 | 0.0% [0.0, 0.0] | 0.006 | 32.1% [24.7, 38.9] | 0.049 |
| SFT (LoRA) | 162 | 0.0% [0.0, 0.0] | 0.012 | 74.1% [67.3, 80.2] | −0.038 |
| GRPO (seed 0) | 162 | 0.0% [0.0, 0.0] | 0.006 | **2.5% [0.6, 5.6]** | **0.113** |

Paired per-puzzle reward: GRPO − SFT = +0.152 [0.133, 0.169]; GRPO − base =
+0.064 [0.046, 0.082]. McNemar on solve rate is degenerate (no arm solves
anything).

## Qwen2.5-7B

| Arm | n | Solve rate | Groups correct | Invalid rate | Mean reward |
|---|---|---|---|---|---|
| base | 162 | 0.0% | 0.160 | 6.8% [3.1, 11.1] | 0.165 |
| SFT (QLoRA) | 162 | **1.2%** (2/162) | **0.346** | 22.2% [16.0, 28.4] | **0.197** |
| GRPO (seed 0) | 162 | 0.0% | 0.025 | **0.6% [0.0, 1.9]** | 0.125 |

Paired SFT − GRPO on groups correct (0-4 scale): +0.296 [0.191, 0.407].

## Reference points (from gvc-local, 8B, multi-agent prompting)

| Arm | n | Solve rate |
|---|---|---|
| gvc-local basic (8B) | 10 | 20.0% [0.0, 50.0] |
| gvc-local GVC multi-agent (8B) | 10 | 60.0% [30.0, 90.0] |

Small n; these are context for the capability ceiling, not matched comparisons.

## Seed replication (3 GRPO seeds per scale)

Each seed re-runs GRPO from the same SFT warm start, isolating RL run-to-run
variance. Same test split, greedy.

| Scale | Metric | seed 0 | seed 1 | seed 2 | mean ± sd |
|---|---|---|---|---|---|
| 7B | groups correct | 0.025 | 0.043 | 0.068 | 0.045 ± 0.022 |
| 7B | invalid rate | 0.006 | 0.019 | 0.012 | 0.012 ± 0.006 |
| 7B | mean reward | 0.125 | 0.129 | 0.141 | 0.132 ± 0.008 |
| 1.5B | groups correct | 0.006 | 0.000 | 0.000 | 0.002 ± 0.004 |
| 1.5B | invalid rate | 0.025 | 0.031 | 0.037 | 0.031 ± 0.006 |
| 1.5B | mean reward | 0.113 | 0.110 | 0.109 | 0.111 ± 0.002 |

Paired SFT − GRPO on groups correct, 7B, per seed: +0.296 [0.191, 0.407],
+0.278 [0.173, 0.389], +0.253 [0.142, 0.370]. All three 7B seeds fall below base
on both grouping (max 0.068 vs 0.160) and reward (max 0.141 vs 0.165).

## pass@16 (temperature 0.9, best-of-k scoring)

Distinguishes capability loss from greedy-decoding degradation.

| Scale | Arm | pass@16 solve | best-of-16 groups correct | any-valid |
|---|---|---|---|---|
| 7B | base | 0.000 [0.000, 0.000] | 0.114 | 0.994 |
| 7B | SFT | **0.043** [0.012, 0.074] | **0.252** | 1.000 |
| 7B | GRPO | 0.012 [0.000, 0.031] | 0.022 | 1.000 |
| 1.5B | base | 0.000 | 0.011 | 0.901 |
| 1.5B | SFT | 0.000 | 0.014 | 0.617 |
| 1.5B | GRPO | 0.000 | 0.002 | 0.988 |

Paired bootstrap on best-of-16 groups correct (7B): SFT − GRPO = +0.920 [+0.772,
+1.062]; base − GRPO = +0.370 [+0.259, +0.475]. Sampling does not recover the
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
policies.** The seed-replication session served four LoRA adapters at once
(different `--max-loras`, different tensor-parallel layout) and re-measured the
SFT arms as a side effect. Comparing the same adapter across the two sessions:

| Arm | main run | seed-eval run | Δ groups correct |
|---|---|---|---|
| 7B SFT | 0.346 | 0.321 | 0.025 |
| 1.5B SFT | 0.012 | 0.019 | 0.007 |
| 7B GRPO seed 0 | 0.025 | 0.025 | **0.000** |
| 1.5B GRPO seed 0 | 0.006 | 0.006 | **0.000** |

Greedy decoding is not bitwise-deterministic across vLLM batching/parallelism
configurations, so a few borderline tokens flip. Practical measurement noise on
`groups_correct` is therefore ~0.025 for SFT arms — an order of magnitude smaller
than the effects claimed here (SFT − GRPO ≈ 0.28-0.32 at 7B), so no conclusion is
affected.

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
