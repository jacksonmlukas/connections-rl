---
license: mit
language:
  - en
library_name: peft
pipeline_tag: text-generation
base_model: Qwen/Qwen2.5-7B-Instruct
base_model_relation: adapter
tags:
  - lora
  - trl
  - peft
  - nyt-connections
  - puzzle-solving
  - qlora
  - sft
  - supervised-fine-tuning
model-index:
  - name: connections-rl-sft-7b
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
            value: 0.012346
          - type: groups_correct
            name: Groups correct (mean, 0-4 scale)
            value: 0.345679
          - type: invalid_rate
            name: Invalid output rate
            value: 0.222222
          - type: reward
            name: Mean reward
            value: 0.196914
---

# connections-rl-sft-7b (7B)

QLoRA supervised fine-tuning adapter for [Qwen2.5-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct) on NYT Connections. **The best-performing arm of the [connections-rl](https://github.com/jacksonmlukas/connections-rl) study and the only one that produces held-out solves.**

> **Reading the numbers.** `Groups correct` is a **mean count on a 0-4 scale**: each board has 4 groups, so a value of 0.346 means 0.346 of 4 groups per board, i.e. 8.6% of groups. The `% of groups` column is that value divided by 4. Values quoted from `results-analysis/` (pass@k, entropy/KL) are already 0-1 fractions.

## Model Details

- **Developed by:** Jackson Lukas
- **Model type:** QLoRA adapter (rank 16, alpha 32, all-linear) for a decoder-only causal LM
- **Base model:** [Qwen/Qwen2.5-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct)
- **Training stage:** Supervised fine-tuning
- **Language:** English
- **License:** MIT (the base model carries its own Qwen license)

### Model Sources

- **Repository:** [https://github.com/jacksonmlukas/connections-rl](https://github.com/jacksonmlukas/connections-rl)
- **Technical report:** [`report/findings.md`](https://github.com/jacksonmlukas/connections-rl/blob/main/report/findings.md)
- **Full result tables:** [`report/results.md`](https://github.com/jacksonmlukas/connections-rl/blob/main/report/results.md)
- **Implementation notes:** [`report/implementation_notes.md`](https://github.com/jacksonmlukas/connections-rl/blob/main/report/implementation_notes.md)
- **Raw eval artifacts:** [`jacksonlukas/connections-rl-results`](https://huggingface.co/datasets/jacksonlukas/connections-rl-results)
- **This adapter:** [`jacksonlukas/connections-rl-sft-7b`](https://huggingface.co/jacksonlukas/connections-rl-sft-7b)

Part of a two-scale, three-seed study of verifiable-reward RL. This adapter is the shared warm start for all three 7B GRPO seed replicates, which is what isolates RL run-to-run variance from SFT variance.

## Intended Uses

### Direct Use

The strongest single-model arm in this study for NYT Connections. It more than doubles the base model's grouping accuracy and is the only arm at any scale to solve held-out boards outright.

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

The 2 of 162 solves are **not statistically significant** (McNemar vs base p = 0.5); they are suggestive, not conclusive. As at 1.5B, SFT increases invalid outputs (6.8% to 22.2%), so it trades grounding for format, though far less severely than at 1.5B. Absolute performance remains far below the 60% reached by an 8B model with multi-agent prompting scaffolding in the predecessor project.

**Study-level limitations that apply to every adapter here:**

- One task (NYT Connections) and one reward design. The conclusion is about this reward's structure/semantics decomposition, not about GRPO in general.
- 807 training boards is small for RL. Memorization is a plausible consequence of data scale as much as of the algorithm.
- Checkpoint-level analyses (entropy, KL, phase transition) come from seed 0 at 7B only, because only that run was Hub-synced during training. Endpoint claims carry n=3 seeds per scale; the timing of the collapse carries n=1.
- NYT Connections boards encode US-centric cultural and idiomatic knowledge, so performance is not representative of word-association ability in general.

### Recommendations

If you want the best grouping ability from this study, use this adapter and expect roughly one in five outputs to be malformed. If you need well-formed output above all, the GRPO adapter achieves 0.6% invalid but destroys grouping.

## How to Get Started

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch

bnb = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.float16,
)
base = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-7B-Instruct", quantization_config=bnb, device_map="auto"
)
tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
model = PeftModel.from_pretrained(base, "jacksonlukas/connections-rl-sft-7b")
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

Rank-16 QLoRA: 4-bit NF4 quantized base with double quantization and fp16
compute, LoRA adapters in fp32, 3 epochs on the 807-puzzle train split. Single
free-tier T4.

Training on a pre-Ampere T4 required working around TRL's unconditional cast of
quantized-model trainable parameters to bf16 (see
[peft#2889](https://github.com/huggingface/peft/issues/2889)), which breaks the
fp16 gradient scaler. The workaround is a post-construction re-cast to fp32 in
[`train/common.py`](https://github.com/jacksonmlukas/connections-rl/blob/main/src/connections_rl/train/common.py).

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

| Metric | base 7B | **SFT** | GRPO (seed 0) |
|---|---|---|---|
| Solve rate | 0.0% | **1.2%** | 0.0% |
| Groups correct (0-4) | 0.160 | **0.346** | 0.025 |
| Groups correct (% of groups) | 4.0% | **8.6%** | 0.6% |
| Invalid outputs | 6.8% | **22.2%** | 0.6% |
| Mean reward | 0.165 | **0.197** | 0.125 |

Scale unlocks genuine partial competence: the base 7B model already solves 4.0% of groups, and SFT more than doubles that to 8.6% while producing the project's first held-out solves. The grounding cost of SFT shrinks with scale but persists.

**Seed replication (7B, 3 GRPO seeds).** Measured in session B:

| Metric | seed 0 | seed 1 | seed 2 | mean ± sd |
|---|---|---|---|---|
| Groups correct (0-4) | 0.025 | 0.043 | 0.068 | 0.045 ± 0.022 |
| Invalid rate | 0.006 | 0.019 | 0.012 | 0.012 ± 0.006 |
| Mean reward | 0.125 | 0.129 | 0.141 | 0.132 ± 0.008 |

Under **pass@16** sampling (temperature 0.9, best-of-k scoring): base 0.000 solve / 0.114 groups (fraction), SFT 0.043 / 0.252, GRPO 0.012 / 0.022.

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
**about 3 hours on a single T4**. The entire 7B study, including all seed
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
