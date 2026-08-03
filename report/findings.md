# GRPO on NYT Connections: what verifiable-reward RL actually transfers

*A two-scale (1.5B / 7B), three-seed study on free-tier GPUs. Sections below run
in the order the work was done: the 1.5B result first, then the 7B ablation that
inverts it, then the replication evidence.*

**TL;DR.** Post-training Qwen2.5 models with GRPO on 807 Connections puzzles drives training reward to its theoretical maximum, but held-out solve rate stays at 0%. What *does* transfer is everything the reward could verify structurally: at 1.5B, invalid outputs (hallucinated words, malformed answer blocks) fall from 74.1% (SFT) to 2.5%, a significant paired reward gain, while grouping ability is memorized rather than learned. At 7B the same recipe reaches the best structural validity in the study (0.6% invalid) but collapses grouping *below the untrained base*, so the trade turns net-harmful. Verifiable-reward RL here is a format-and-grounding teacher, not a reasoning teacher — and whether that is worth having depends on how much semantic ability the starting policy had. Replicated over 3 seeds per scale, with behavioral, dynamical, and weight-space evidence.

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

- **Pre-Ampere dtype failures with TRL GRPO:** four distinct modes documented — FSDP bf16/fp32 flatten mismatch from fp32 PEFT layers; the emulated-bf16 `is_bf16_supported()` trap on T4 (returns True, then fails); fp16 mixed precision vs `generate()` rollouts under FSDP; and TRL's *unconditional* cast of QLoRA trainable params to bf16 inside `SFTTrainer`/`GRPOTrainer.__init__` (no opt-out, per peft#2889), which breaks the fp16 GradScaler on any pre-Ampere GPU. Single-GPU with a post-construction re-cast to fp32 was the only stable path. See `src/connections_rl/train/common.py::fix_qlora_adapter_dtype_for_pre_ampere`.
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

## Is it one bad seed? No: three levels of replication

The obvious objection to a single RL run is variance. Two extra GRPO seeds per scale (six runs total, each re-run from the *same* SFT warm start so that only RL run-to-run variance varies) answer it three ways.

**Behavioral (held-out test, greedy, n=162).**

| Scale | Metric | seed 0 | seed 1 | seed 2 | mean ± sd |
|---|---|---|---|---|---|
| 7B | groups correct | 0.025 | 0.043 | 0.068 | 0.045 ± 0.022 |
| 7B | invalid rate | 0.006 | 0.019 | 0.012 | 0.012 ± 0.006 |
| 7B | mean reward | 0.125 | 0.129 | 0.141 | 0.132 ± 0.008 |
| 1.5B | groups correct | 0.006 | 0.000 | 0.000 | 0.002 ± 0.004 |
| 1.5B | invalid rate | 0.025 | 0.031 | 0.037 | 0.031 ± 0.006 |
| 1.5B | mean reward | 0.113 | 0.110 | 0.109 | 0.111 ± 0.002 |

Every 7B seed lands below base on grouping (max 0.068 vs 0.160) and below base on reward (max 0.141 vs 0.165); paired bootstrap of SFT − GRPO on groups correct yields +0.296 [0.191, 0.407], +0.278 [0.173, 0.389], +0.253 [0.142, 0.370]. Every 1.5B seed holds invalid near 3% with reward above base. Seed 0 — the originally published run — is the least favorable 7B draw on grouping, so the headline table understates GRPO rather than flattering it.

**Dynamical.** The checkpoint decomposition (7B, val split) shows the mechanism rather than only the endpoint: through step 100 GRPO improves *both* components (structure 0.79 → 0.93, semantics 0.102 → 0.139, reward peaking at 0.289); between steps 100 and 150 semantics collapses an order of magnitude (0.120 → 0.012) while structure locks at 0.97 and never moves again. Mean reward *falls* from its step-50 peak and plateaus below it, so the collapse is not even reward-optimal on held-out data — the policy found a structural local optimum on the training distribution and stopped exploring.

**Parametric.** Independent seeds move the policy in the *same direction* in weight space. Cosine similarity between seeds' RL-induced effective LoRA updates (dW = B_grpo A_grpo − B_sft A_sft) is +0.68 to +0.69 at 7B and +0.78 to +0.80 at 1.5B, against a random-direction expectation of ~1e−5 in a 6.5-billion-dimensional update space. The control cos(dW_RL, dW_SFT) is ≈0 at both scales, so the agreement is not inherited from the shared warm start. Update magnitudes agree within 4%, and the largest per-module changes concentrate in the same mid-layer MLP `up_proj`/`gate_proj` blocks (layers 9-18) at both scales, with per-module cosines of 0.80-0.91.

Taken together: the semantic collapse is a systematic attractor of this reward under this optimizer, reproducible across seeds and scales, localized to the same parameters, and visible as a phase transition during training. Remaining limitations are one task and one reward design, not one run.

**Reproducibility note.** Independent of seeds, the published 1.5B eval was re-run end-to-end on a fresh VM under the same serving configuration and reproduced byte-identically (greedy decoding). Under a *different* vLLM configuration (four adapters served at once), greedy results drift slightly for SFT arms (Δ groups correct ≈ 0.025 at 7B) but not at all for GRPO arms, which reproduce exactly — an independent corroboration of entropy collapse, since a policy at entropy ~3e−4 has no borderline token decisions left to flip. Measurement noise is an order of magnitude below the effects claimed. Details in [`results.md`](results.md).

## What would move solve rate

Larger base model (7–8B, where gvc-local shows nonzero single-model competence), reasoning-style completions (current outputs are ~75 tokens — near-zero deliberation; reward shaping or length incentives could force chain-of-thought), and more training signal per board (paraphrased/shuffled board augmentation to break memorization).
