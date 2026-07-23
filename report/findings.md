# GRPO on NYT Connections at 1.5B: what verifiable-reward RL actually transfers

**TL;DR.** Post-training Qwen2.5-1.5B-Instruct with GRPO on 807 Connections puzzles drives training reward to its theoretical maximum, but held-out solve rate stays at 0%. What *does* transfer is everything the reward could verify structurally: invalid outputs (hallucinated words, malformed answer blocks) fall from 74.1% (SFT) to 2.5%, a significant paired reward gain. The grouping ability itself was memorized, not learned. At this scale, verifiable-reward RL is a format-and-grounding teacher, not a reasoning teacher.

## Setup

- **Model:** Qwen2.5-1.5B-Instruct, rank-16 LoRA. SFT warm start (3 epochs on train split), then GRPO (TRL GRPOTrainer, K=8 completions/puzzle, group-relative advantage, KL to the SFT reference).
- **Data:** 1,078 puzzles from gvc-local, strictly chronological split — train n=807 (2023-06-12 → 2025-08-27), val n=108, test n=162 (2025-12-15 → 2026-05-29). Nothing tested on predates anything trained on.
- **Reward (deterministic, unit-tested):** +0.1 format validity (all 16 board words, 4×4, each once), +1.0 × (correct groups / 4), +0.5 solve bonus, +0.05 one-away credit, −0.1 invalid. Max = 1.6.
- **Compute:** $0 — Colab T4 and Kaggle T4s. GRPO v2: single T4, ~10 h, 806 steps (2 epochs), lr 5e-6, KL β=0.001, temperature 0.9.

## Results (test, n=162, bootstrap 95% CIs)

| Arm | Solve rate | Invalid rate | Mean reward |
|---|---|---|---|
| base | 0.0% | 32.1% [24.7, 38.9] | 0.049 |
| SFT | 0.0% | 74.1% [67.3, 80.2] | −0.038 |
| GRPO | 0.0% | 2.5% [0.6, 5.6] | 0.113 |

Paired per-puzzle reward differences: GRPO−SFT = +0.152 [0.133, 0.169]; GRPO−base = +0.064 [0.046, 0.082]. McNemar on solve rate is degenerate (no arm solves anything; p=1.0).

## Two hyperparameter regimes, two failure modes

**v1 (lr 1e-6, KL β=0.04, 1 epoch):** the policy never moved. KL stayed ≈0.001 for 403 steps; greedy decoding produced outputs byte-identical to SFT. Diagnosis: the KL penalty and LR jointly over-constrained the update — a common silent failure when copying "safe" PPO-era defaults into GRPO.

**v2 (lr 5e-6, KL β=0.001 — the DeepSeek-R1 value — 2 epochs):** training reward climbed from ≈0 to the 1.6 maximum by step ~270 and pinned there; reward std → 0; entropy → 3e-4. The policy became a deterministic lookup table over training boards. Held-out grouping accuracy: 0.6% of groups.

## Interpretation

The reward has two components of very different learnability:

1. **Structural validity** (parseable `<ANSWER>` block, exactly the 16 board words, 4×4 partition) is verifiable *per-sample* and generalizes as a policy: "only emit words you see on the board." GRPO learned it essentially perfectly, fixing SFT's dominant failure mode (SFT *increased* hallucination to 74% by teaching the answer format without grounding).
2. **Semantic grouping** requires world knowledge and wordplay that a 1.5B model largely lacks. With only 807 training boards and a saturating reward, the shortest descent path is memorization — which the entropy collapse makes visible in the training curve alone.

Reference points from gvc-local (8B, multi-agent prompting): 20% basic, 60% multi-agent. The capability gap between 1.5B+RL and 8B+scaffolding is not closable by this training recipe; the model, not the optimizer, is the binding constraint.

## Negative results worth keeping

- **FSDP on 2×T4 with TRL GRPO:** three distinct dtype failures documented (bf16/fp32 flatten mismatch from fp32 PEFT layers; emulated-bf16 `is_bf16_supported()` trap on T4; fp16 mixed precision vs `generate()` rollouts). Single-GPU was the only stable path on pre-Ampere hardware.
- **SFT alone is harmful here:** it teaches the format but not board-grounding, raising invalid rate from 32% → 74%. RL against the verifiable reward is what repairs it.

## Scale ablation: the finding sharpens at 7B

Rerunning the identical pipeline on Qwen2.5-7B-Instruct (QLoRA on the same T4s; SFT 3 epochs, GRPO 1 epoch with the v2 hypers) inverts the value of RL:

| Arm (7B) | Solve | Groups correct | Invalid | Mean reward |
|---|---|---|---|---|
| base | 0.0% | 0.160 | 6.8% | 0.165 |
| SFT | 1.2% (2/162) | 0.346 | 22.2% | 0.197 |
| GRPO | 0.0% | 0.025 | 0.6% | 0.125 |

Three observations. First, scale unlocks genuine partial competence: base 7B gets 16% of groups with 6.8% invalid (1.5B: 0.6% and 32.1%), and SFT doubles grouping to 34.6% while producing the first held-out solves of the project (2/162; McNemar vs base p=0.5 — suggestive, not significant). Second, SFT's grounding cost shrinks with scale but persists (invalid 32→74% at 1.5B; 6.8→22.2% at 7B). Third — the sharpest result — **GRPO is net-harmful at 7B**: it reaches the best structural validity of any arm at any scale (0.6% invalid) while collapsing grouping *below the untrained base* (0.025 vs 0.160), leaving mean reward under base (0.125 vs 0.165). The policy converged to "emit a perfectly valid partition" and traded away semantics to get there.

The cross-scale conclusion is cleaner than either run alone: GRPO against a structurally-verifiable reward optimizes exactly what the reward can verify, at the expense of what it can't. At 1.5B the starting policy had no semantic ability to lose, so the trade was pure gain (hallucination fixed). At 7B it had real ability, and the same optimization destroyed it. The failure is not model scale — it is that the reward's cheap-to-verify component (structure) and expensive-to-verify component (semantics) decouple under optimization pressure with only 807 training boards.

## What would move solve rate

Larger base model (7–8B, where gvc-local shows nonzero single-model competence), reasoning-style completions (current outputs are ~75 tokens — near-zero deliberation; reward shaping or length incentives could force chain-of-thought), and more training signal per board (paraphrased/shuffled board augmentation to break memorization).
