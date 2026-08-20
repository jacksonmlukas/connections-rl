# TASK D runbook — step-50 checkpoint on the TEST split (owner-launched, ~1 h GPU)

One vLLM session serves all three arms so every comparison is within-session
(guardrail 2). Nothing here touches existing results; all outputs land under
`results-analysis/aug20/taskD-session/` (guardrail 3).

## 0. Session setup (Kaggle/Colab T4, or any CUDA box)

```bash
git clone https://github.com/jacksonmlukas/connections-rl && cd connections-rl
git checkout analysis/aug20
pip install -e ".[dev]" && pip install vllm
export CONNECTIONS_PUZZLES=path/to/gvc-local/data/puzzles/tagged_connections.json
make data                        # rebuilds data/splits deterministically
huggingface-cli login            # your token, your side only
pip show vllm | awk '/^Version/{print}' > results-analysis/aug20/taskD-session-vllm-version.txt
mkdir -p results-analysis/aug20/taskD-session
```

## 1. Fetch the two adapters

```bash
# step-50 checkpoint (private ckpt repo; save_steps=50 in grpo-7b.yaml means
# the step-50 sync is the earliest checkpoint in the repo)
huggingface-cli download jacksonlukas/connections-rl-grpo-7b-ckpt \
    --local-dir adapters/grpo-7b-ckpt
ls adapters/grpo-7b-ckpt          # confirm a checkpoint-50/ directory exists;
                                  # if the layout differs, STOP and report it
# final seed-0 adapter (same one the published results used)
huggingface-cli download jacksonlukas/connections-rl-grpo-7b \
    --local-dir adapters/grpo-7b
```

## 2. Serve all three arms in ONE vLLM session

```bash
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-7B-Instruct \
    --enable-lora --max-loras 2 --max-lora-rank 16 \
    --lora-modules \
      connections-rl-grpo-7b-ckpt50=adapters/grpo-7b-ckpt/checkpoint-50 \
      connections-rl-grpo-7b=adapters/grpo-7b \
    --port 8000
```

## 3. Evaluate (same harness, greedy, 162-puzzle test split)

```bash
python -m connections_rl.eval.run --config results-analysis/aug20/taskD_eval_ckpt50.yaml
```

Writes `taskD-session/{base,grpo-ckpt50,grpo-final}/{metrics.json,records.jsonl}`
plus `taskD-session/comparisons.json` (McNemar + solve-rate diff CIs, the
harness's standard pairs).

## 4. Paired bootstrap on groups-correct (repo estimator, seed 0)

```bash
python results-analysis/aug20/taskD_paired_groups.py
```

Writes `taskD-session/paired_groups.json` with ckpt50−base, final−base, and
ckpt50−final on the 0-4 scale.

## 5. Sanity checks before reading anything into the numbers

- `grpo-final` here should be close to, but need NOT equal, session A's
  0.025/0.6%/0.125 — this is a NEW session; GRPO arms have reproduced exactly
  across sessions before, so a nonzero difference on GRPO would itself be
  notable. Do not "fix" any difference (guardrail 1).
- The base arm should be close to 0.160 groups; small drift vs session A is
  expected for non-GRPO arms.
- If any download or the checkpoint-50 layout fails, stop and report; do not
  substitute a different checkpoint.

---

# TASK F runbook — β=0.01 run (OPTIONAL, only after Task D completes)

Roughly 5.5 T4-hours of training + eval; fits after Task D inside a 12 h cap.

```bash
python -m connections_rl.train.grpo --config results-analysis/aug20/taskF_grpo-7b-beta01.yaml
```

The config is identical to `configs/train/grpo-7b.yaml` except `kl_beta: 0.01`,
with new `output_dir`, `ckpt_hub_repo` (`connections-rl-grpo-7b-beta01-ckpt`),
and `run_name`, so nothing existing is overwritten. Checkpoints sync every 50
steps as the original did.

After training, evaluate the final adapter on the test split (add it as a
fourth `--lora-modules` entry and a fourth arm in a copy of the Task D config
pointing at a NEW out_dir, e.g. `taskF-session/`), then the checkpoint
trajectory in the same JSON shape as `entropy-kl-7b.json`:

```bash
python -m connections_rl.eval.entropy_kl \
    --model Qwen/Qwen2.5-7B-Instruct --load-in-4bit \
    --sft-adapter artifacts/sft-7b \
    --checkpoints base=base sft=artifacts/sft-7b \
      ckpt-50=artifacts/grpo-7b-beta01/checkpoint-50 \
      ckpt-100=artifacts/grpo-7b-beta01/checkpoint-100 \
      ckpt-150=artifacts/grpo-7b-beta01/checkpoint-150 \
      ckpt-200=artifacts/grpo-7b-beta01/checkpoint-200 \
      ckpt-250=artifacts/grpo-7b-beta01/checkpoint-250 \
      ckpt-300=artifacts/grpo-7b-beta01/checkpoint-300 \
      ckpt-350=artifacts/grpo-7b-beta01/checkpoint-350 \
      ckpt-403=artifacts/grpo-7b-beta01/checkpoint-403 \
    --puzzles data/splits/puzzles_val.json --n 100 --temperature 0.9 \
    --out results-analysis/aug20/entropy-kl-7b-beta01
```

(Adjust the checkpoint list to whatever steps actually exist; exact flag names
per `src/connections_rl/eval/entropy_kl.py` argparse — verify `--model`,
`--sft-adapter`, `--load-in-4bit` spellings against the file before running.)

**If the run dies overnight: report that it died and stop. One retry maximum.
Tasks A–E are already delivered and are worth more than this run.**
