---
base_model: Qwen/Qwen2.5-1.5B-Instruct
library_name: peft
license: mit
tags:
  - lora
  - sft
  - trl
  - nyt-connections
  - puzzle-solving
---

# connections-rl-sft (Qwen2.5-1.5B)

Rank-16 LoRA SFT adapter for [Qwen2.5-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct), trained to solve NYT Connections puzzles. The SFT warm start for the GRPO adapter [connections-rl-grpo](https://huggingface.co/jacksonlukas/connections-rl-grpo).

Part of [**connections-rl**](https://github.com/jacksonmlukas/connections-rl): a two-scale study of what verifiable-reward RL actually transfers, built and trained entirely on free-tier GPUs (Colab/Kaggle T4s).

## Training

3 epochs of LoRA SFT (r=16, α=32, all-linear) on 807 puzzles (2023-06 → 2025-08), chat-formatted with a fixed `<ANSWER>` output schema. Chronological split: everything evaluated on postdates everything trained on.

## Evaluation (162 held-out puzzles, 2025-12 → 2026-05)

| Metric | base | **this adapter** | GRPO |
|---|---|---|---|
| Solve rate | 0.0% | 0.0% | 0.0% |
| Invalid outputs | 32.1% | **74.1%** | 2.5% |
| Mean reward | 0.049 | −0.038 | 0.113 |

**Honest caveat: this adapter makes hallucination *worse*.** SFT teaches the answer format without board-grounding — it more than doubles the rate of answers containing words not on the board. That failure mode is what the GRPO stage repairs. Full analysis in the [technical report](https://github.com/jacksonmlukas/connections-rl/blob/main/report/findings.md).

## Usage

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM

base = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")
model = PeftModel.from_pretrained(base, "jacksonlukas/connections-rl-sft")
```

Or serve with vLLM: `vllm serve Qwen/Qwen2.5-1.5B-Instruct --enable-lora --lora-modules connections-rl-sft=<path>`
