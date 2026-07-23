---
base_model: Qwen/Qwen2.5-7B-Instruct
library_name: peft
license: mit
tags:
  - lora
  - qlora
  - sft
  - trl
  - nyt-connections
---

# connections-rl-sft-7b (Qwen2.5-7B)

QLoRA SFT adapter for [Qwen2.5-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct) on NYT Connections — **the best-performing arm of the [connections-rl](https://github.com/jacksonmlukas/connections-rl) study**, producing its only held-out solves.

## Training

Rank-16 QLoRA (4-bit NF4 base, fp16 compute), 3 epochs on 807 puzzles, trained on a free Kaggle T4. Chronological split: all evaluation puzzles postdate all training puzzles.

## Evaluation (162 held-out puzzles, 2025-12 → 2026-05)

| Metric | base 7B | **this adapter** | GRPO 7B |
|---|---|---|---|
| Solve rate | 0.0% | **1.2%** (2/162) | 0.0% |
| Groups correct (mean) | 0.160 | **0.346** | 0.025 |
| Invalid outputs | 6.8% | 22.2% | 0.6% |
| Mean reward | 0.165 | **0.197** | 0.125 |

Doubles the base model's grouping accuracy and yields the first held-out solves of the project (not statistically significant at n=162; McNemar p=0.5). Note the trade-off: like at 1.5B, SFT increases invalid outputs (format without grounding), though far less severely. The follow-on GRPO stage eliminated invalid outputs but destroyed grouping ability — see the [technical report](https://github.com/jacksonmlukas/connections-rl/blob/main/report/findings.md).

## Usage

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM

base = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
model = PeftModel.from_pretrained(base, "jacksonlukas/connections-rl-sft-7b")
```
