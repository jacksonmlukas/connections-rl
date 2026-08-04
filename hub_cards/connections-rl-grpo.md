---
license: mit
language:
  - en
library_name: peft
pipeline_tag: text-generation
base_model: Qwen/Qwen2.5-1.5B-Instruct
base_model_relation: adapter
tags:
  - lora
  - trl
  - peft
  - nyt-connections
  - puzzle-solving
  - grpo
  - reinforcement-learning
  - rlvr
  - reward-over-optimization
  - negative-results
model-index:
  - name: connections-rl-grpo
    results:
      - task:
          type: text-generation
          name: NYT Connections puzzle solving
        dataset:
          type: custom
          name: NYT Connections held-out test split (chronological, n=162)
          split: test
        metrics:
          - type: accuracy
            name: Solve rate
            value: 0.000000
          - type: groups_correct
            name: Groups correct (mean, 0-4 scale)
            value: 0.006173
          - type: invalid_rate
            name: Invalid output rate
            value: 0.024691
          - type: reward
            name: Mean reward
            value: 0.113272
---

# connections-rl-grpo (1.5B)

GRPO (verifiable-reward RL, DeepSeek-R1 style) LoRA adapter for [Qwen2.5-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct), warm-started from [connections-rl-sft](https://huggingface.co/jacksonlukas/connections-rl-sft).

**Headline result: RL transferred exactly what the reward could verify.** Invalid outputs fall from 74.1% (SFT) and 32.1% (base) to **2.5%**, a significant paired reward gain, while grouping ability does not generalize at all.

> **Reading the numbers.** `Groups correct` is a **mean count on a 0-4 scale**: each board has 4 groups, so the SFT value of 0.012 below means 0.012 of 4 groups per board, i.e. 0.3% of groups. The `% of groups` column is that value divided by 4. **Invalid rate** is shown as a percentage in the arm-comparison table and as a bare 0-1 fraction in the seed table, matching how each is stored; both rows are labeled with their units. Values quoted from `results-analysis/` (pass@k, entropy/KL) are already 0-1 fractions.

## Model Details

- **Developed by:** Jackson Lukas
- **Model type:** LoRA adapter (rank 16, alpha 32, all-linear) for a decoder-only causal LM
- **Base model:** [Qwen/Qwen2.5-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct)
- **Training stage:** GRPO (verifiable-reward RL) on top of the SFT warm start
- **Language:** English
- **License:** MIT (the base model carries its own Qwen license)

### Model Sources

- **Repository:** [https://github.com/jacksonmlukas/connections-rl](https://github.com/jacksonmlukas/connections-rl)
- **Technical report:** [`report/findings.md`](https://github.com/jacksonmlukas/connections-rl/blob/main/report/findings.md)
- **Full result tables:** [`report/results.md`](https://github.com/jacksonmlukas/connections-rl/blob/main/report/results.md)
- **Implementation notes:** [`report/implementation_notes.md`](https://github.com/jacksonmlukas/connections-rl/blob/main/report/implementation_notes.md)
- **Raw eval artifacts:** [`jacksonlukas/connections-rl-results`](https://huggingface.co/datasets/jacksonlukas/connections-rl-results)
- **This adapter:** [`jacksonlukas/connections-rl-grpo`](https://huggingface.co/jacksonlukas/connections-rl-grpo)

Part of [**connections-rl**](https://github.com/jacksonmlukas/connections-rl), a two-scale, three-seed study. The companion 7B adapter shows the *same* recipe turning net-harmful once the starting policy has real semantic ability to lose.

## Intended Uses

### Direct Use

A worked example of verifiable-reward RL acting as a format-and-grounding teacher. The adapter reliably emits well-formed 4x4 partitions using only words present on the board, which is the behavior the reward could check per sample.

### Downstream Use

The adapter and the surrounding harness are intended as a reproducible artifact for research on reward design and reward over-optimization in verifiable-reward RL. The reward, the leakage-aware splits, the evaluation harness with bootstrap CIs and paired tests, and every checkpoint are public so the finding can be re-derived or contradicted.

### Out-of-Scope Use

This is a research artifact, not a product. Do not use it to:

- solve live NYT Connections puzzles competitively or to build a puzzle-solving
  service; held-out solve rate is at or near zero for every arm in this study;
- draw conclusions about the base model's general capability, since these
  adapters are narrowly specialized and at least one of them measurably degrades
  the base model's ability on this task;
- perform any high-stakes reasoning task. Nothing here was evaluated for safety,
  toxicity, factuality or robustness outside the Connections task.

## Bias, Risks, and Limitations

Solve rate is **0 of 162** held-out puzzles, and grouping accuracy does not improve over base. Training reward saturated at its theoretical maximum (1.6) with reward variance exactly 0 and policy entropy near 0, meaning the policy memorized the 807 training answers rather than learning to group. The gain is confined to output validity.

**Study-level limitations that apply to every adapter here:**

- One task (NYT Connections) and one reward design. The conclusion is about this reward's structure/semantics decomposition, not about GRPO in general.
- 807 training boards is small for RL. Memorization is a plausible consequence of data scale as much as of the algorithm.
- Checkpoint-level analyses (entropy, KL, phase transition) come from seed 0 at 7B only, because only that run was Hub-synced during training. Endpoint claims carry n=3 seeds per scale; the timing of the collapse carries n=1.
- NYT Connections boards encode US-centric cultural and idiomatic knowledge, so performance is not representative of word-association ability in general.

### Recommendations

Treat the 2.5% invalid rate as the real result and the 0% solve rate as the equally real limitation. Do not extrapolate the format gain into a claim about reasoning.

## How to Get Started

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

base = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-1.5B-Instruct", dtype=torch.float16, device_map="auto"
)
tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")
model = PeftModel.from_pretrained(base, "jacksonlukas/connections-rl-grpo")
model.eval()

words = ["HAIL", "RAIN", "SLEET", "SNOW", "BUCKS", "HEAT", "JAZZ", "NETS",
         "OPTION", "RETURN", "SHIFT", "TAB", "KAYAK", "LEVEL", "MOM", "RACECAR"]
prompt = tok.apply_chat_template(
    [{"role": "user", "content": "Group these 16 words into 4 groups of 4:\n"
       + ", ".join(words)}],
    tokenize=False, add_generation_prompt=True,
)
out = model.generate(**tok(prompt, return_tensors="pt").to(model.device),
                     max_new_tokens=256, do_sample=False)
print(tok.decode(out[0], skip_special_tokens=True))
```

The exact prompt template, parser and reward used in the paper are in
[`src/connections_rl`](https://github.com/jacksonmlukas/connections-rl/tree/main/src/connections_rl). Serving the adapter
behind vLLM (`--enable-lora`) is supported via `docker compose up` in the repo.

## Training Details

### Training Data

1,078 NYT Connections puzzles from the [gvc-local](https://github.com/jacksonmlukas/gvc-local)
tagged puzzle database, split **strictly chronologically** so that every evaluation
puzzle postdates every training puzzle:

| Split | n | Date range |
|---|---|---|
| train | 807 | 2023-06-12 to 2025-08-27 |
| val | 108 | 2025-08-28 to 2025-12-14 |
| test | 162 | 2025-12-15 to 2026-05-29 |

The chronological split is the leakage control: NYT Connections boards are
published daily and widely discussed online, so a random split would let a model
benefit from puzzles whose answers circulated before its pre-training cutoff.

### Reward Function

Deterministic and unit-tested, with no learned reward model:

| Component | Value |
|---|---|
| Format validity (all 16 board words, 4x4, each used once) | +0.1 |
| Correct groups | +1.0 x (correct / 4) |
| Full solve bonus | +0.5 |
| One-away shaping | +0.05 |
| Invalid output penalty | -0.1 |
| **Maximum** | **1.6** |

Only the structural component is cheaply verifiable per sample. The study's
central finding is that GRPO optimizes that component at the expense of the
semantic component, which is the part the reward cannot check directly.

### Training Procedure

TRL `GRPOTrainer` over the SFT warm start. K=8 completions per puzzle,
group-relative advantage, KL penalty to the frozen SFT reference.

| Hyperparameter | Value |
|---|---|
| `num_generations` (K) | 8 |
| `temperature` | 0.9 |
| `beta` (KL to SFT reference) | 0.001 |
| `learning_rate` | 5e-6 |
| `per_device_train_batch_size` | 2 |
| `gradient_accumulation_steps` | 8 |
| `num_train_epochs` | 2 |
| `max_completion_length` | 512 |

`scale_rewards`, `loss_type`, `num_iterations` and the clipping epsilons were not
set explicitly and therefore inherited TRL defaults (`scale_rewards="group"`,
`loss_type="dapo"`). This is recorded deliberately, because `scale_rewards="group"`
is the normalization Dr. GRPO ([2503.20783](https://huggingface.co/papers/2503.20783))
argues against. See [`implementation_notes.md`](https://github.com/jacksonmlukas/connections-rl/blob/main/report/implementation_notes.md).

Training reward reached the 1.6 maximum by roughly step 270, at which point `frac_reward_zero_std` reached 1.0: every generation group was saturated, advantages were identically zero, and the final third of training produced no gradient signal.

## Evaluation

### Testing Data, Factors and Metrics

Held-out test split of **162 puzzles (2025-12-15 to 2026-05-29)**, strictly after
every training date. Greedy decoding via vLLM. Bracketed values are percentile
bootstrap 95% CIs over 1,000 resamples. Comparisons between arms use exact
McNemar tests on solve rate and paired bootstrap on per-puzzle reward. Results
are stratified by puzzle category (wordplay, cultural, category, tag-fillin,
silent-letter) in the underlying JSON.

Numbers below come from **session A (main run)**. See
[Reproducibility](#reproducibility-and-measurement-noise).

### Results

| Metric | base 1.5B | SFT | **GRPO (seed 0)** |
|---|---|---|---|
| Solve rate | 0.0% | 0.0% | **0.0%** |
| Groups correct (0-4) | 0.006 | 0.012 | **0.006** |
| Groups correct (% of groups) | 0.2% | 0.3% | **0.2%** |
| Invalid outputs (%) | 32.1% | 74.1% | **2.5%** |
| Mean reward | 0.049 | -0.038 | **0.113** |

*The SFT baseline above is the **session A** measurement (groups correct 0.012). The seed-replicate cards report 0.019 for the same adapter, measured in session B. Both are correct: greedy decoding is not bitwise deterministic across vLLM layouts. See [Reproducibility](#reproducibility-and-measurement-noise) below.*

The reward has two components of very different learnability. Structural validity is verifiable per sample and generalizes as a policy: emit only words on the board. Semantic grouping requires knowledge a 1.5B model largely lacks, so with 807 boards the shortest descent path is memorization. Paired per-puzzle reward differences: GRPO minus SFT = +0.152 [0.133, 0.169]; GRPO minus base = +0.064 [0.046, 0.082].

**Seed replication (1.5B, 3 GRPO seeds).** Measured in session B:

| Metric | seed 0 | seed 1 | seed 2 | mean ± sd |
|---|---|---|---|---|
| Groups correct (0-4) | 0.006 | 0.000 | 0.000 | 0.002 ± 0.004 |
| Invalid rate (0-1 fraction) | 0.025 | 0.031 | 0.037 | 0.031 ± 0.006 |
| Mean reward | 0.113 | 0.110 | 0.109 | 0.111 ± 0.002 |

Under **pass@16** sampling (temperature 0.9, best-of-k scoring): base 0.000 solve / 0.011 groups (fraction), SFT 0.000 / 0.014, GRPO 0.000 / 0.002.

## Reproducibility and Measurement Noise

Some arms were measured in two vLLM serving configurations, referred to in the
repository as session A (`results/`, `results-7b/`) and session B
(`results-seeds-*/`, aggregated in `results-seeds/seed_summary.json`).

Greedy decoding is not bitwise deterministic across vLLM batching and
parallelism layouts, so a small number of borderline tokens flip. The 7B SFT arm
differs on 2 of 162 puzzles for grouping and 3 of 162 for format validity between
sessions; **every GRPO arm reproduces exactly (0 of 162 on all metrics)**.

That asymmetry is itself evidence for the entropy-collapse account: the final 7B
GRPO policy has a measured entropy of 0.0099 nats/token and therefore has no
borderline decisions left to flip, whereas the SFT policy has the highest entropy
of any arm (0.303 nats/token). Statistics are never mixed across sessions within
a single comparison.

## Environmental Impact

Trained on free-tier NVIDIA T4 GPUs (16 GB, Turing, pre-Ampere) via Kaggle and
Google Colab. Total training compute for this adapter was approximately
**about 10 hours on a single T4**. The entire 1.5B study, including all seed
replicates and evaluation, was run at $0 marginal cost on free-tier hardware.
Carbon emissions were not directly measured; the T4 has a 70 W TDP, which bounds
the energy use of a single run well below 1 kWh.

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

## Model Card Contact

Open an issue at [https://github.com/jacksonmlukas/connections-rl/issues](https://github.com/jacksonmlukas/connections-rl/issues).
