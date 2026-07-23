---
base_model: Qwen/Qwen2.5-7B-Instruct
library_name: peft
license: mit
tags:
  - lora
  - qlora
  - grpo
  - trl
  - reinforcement-learning
  - nyt-connections
---

# connections-rl-grpo-7b (Qwen2.5-7B)

GRPO (verifiable-reward RL) QLoRA adapter for [Qwen2.5-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct), warm-started from [connections-rl-sft-7b](https://huggingface.co/jacksonlukas/connections-rl-sft-7b).

**This adapter is published as a documented negative result.** It achieves the best structural validity of any arm at any scale in the [connections-rl](https://github.com/jacksonmlukas/connections-rl) study (0.6% invalid outputs) — while collapsing semantic grouping ability *below the untrained base model*. It is a clean, measured example of reward over-optimization: the policy converged to "emit a perfectly valid partition" and traded away semantics to get there.

## Training

TRL `GRPOTrainer` over the QLoRA SFT adapter; K=8, KL β=0.001, lr 5e-6, 1 epoch on 807 puzzles (free Kaggle T4, ~5.5h, checkpoint-synced to the Hub). Training reward saturated with entropy → 0 (memorization), as at 1.5B.

## Evaluation (162 held-out puzzles, chronological split)

| Metric | base 7B | SFT 7B | **this adapter** |
|---|---|---|---|
| Solve rate | 0.0% | 1.2% | 0.0% |
| Groups correct (mean) | 0.160 | 0.346 | 0.025 |
| Invalid outputs | 6.8% | 22.2% | **0.6%** |
| Mean reward | 0.165 | 0.197 | 0.125 |

Cross-scale conclusion: GRPO against a structurally-verifiable reward optimizes what the reward can verify (structure) at the expense of what it can't (semantics). At 1.5B — where the starting policy had no semantic ability to lose — the identical recipe was a pure win. Full analysis in the [technical report](https://github.com/jacksonmlukas/connections-rl/blob/main/report/findings.md).

## Usage

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM

base = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
model = PeftModel.from_pretrained(base, "jacksonlukas/connections-rl-grpo-7b")
```
