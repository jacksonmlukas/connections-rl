---
base_model: Qwen/Qwen2.5-1.5B-Instruct
library_name: peft
license: mit
tags:
  - lora
  - grpo
  - trl
  - reinforcement-learning
  - nyt-connections
---

# connections-rl-grpo (Qwen2.5-1.5B)

GRPO (verifiable-reward RL, DeepSeek-R1 style) LoRA adapter for [Qwen2.5-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct), warm-started from [connections-rl-sft](https://huggingface.co/jacksonlukas/connections-rl-sft) and trained on NYT Connections puzzles.

Part of [**connections-rl**](https://github.com/jacksonmlukas/connections-rl): a two-scale study of what verifiable-reward RL actually transfers, built and trained entirely on free-tier GPUs.

## Training

TRL `GRPOTrainer`, K=8 completions/puzzle, group-relative advantage, KL penalty (β=0.001) to the SFT reference; lr 5e-6, 2 epochs over 807 puzzles, single T4 (~10h). Deterministic reward: format validity + correct groups + solve bonus − invalid penalty. Training reward saturated at the theoretical maximum (1.6) with policy entropy collapsing to ~0 — i.e. the policy memorized its training answers.

## Evaluation (162 held-out puzzles, chronological split)

| Metric | base | SFT | **this adapter** |
|---|---|---|---|
| Solve rate | 0.0% | 0.0% | 0.0% |
| Invalid outputs | 32.1% | 74.1% | **2.5%** |
| Mean reward | 0.049 | −0.038 | **0.113** |

**Key finding: RL transferred exactly what the reward could verify.** Grouping ability did not generalize (memorization), but format discipline and board-grounding did — invalid outputs dropped 74.1% → 2.5% vs SFT, a significant paired reward gain (+0.152, 95% CI [0.133, 0.169]). See the [technical report](https://github.com/jacksonmlukas/connections-rl/blob/main/report/findings.md) — including the 7B ablation where this same trade turns net-harmful.

## Usage

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM

base = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")
model = PeftModel.from_pretrained(base, "jacksonlukas/connections-rl-grpo")
```
