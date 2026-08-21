# aug21 runbook — post-review experiments (B1/B2/C1.1/C2 GPU bundle + B3)

Branch `analysis/aug21`. New outputs only; nothing existing modified. All the
hard guardrails from the first handoff stand, especially: the two-session
difference is expected and is not to be "fixed."

## 0. On the Mac, before anything (5 min)

1. **Apply the B1 patch.** `src/connections_rl/train/grpo.py` builds GRPOConfig
   from a fixed kwargs dict — a `scale_rewards:` key in YAML is silently
   ignored without the three insertions in `taskB1_grpo_scale_rewards.patch`
   (conditional kwarg + signature guard + **read-back verification off the
   constructed GRPOConfig**, which aborts on mismatch and prints the resolved
   value plus the TRL version — R3a). Apply by hand (~20 lines), commit on
   `analysis/aug21`, push. **Without this, B1 replicates the original run and
   reports it as an ablation.** The notebook's cell 2 applies the same three
   insertions in-session if the branch copy is missing any of them, and cell
   2b runs the handoff's dry check — do not launch training until it passes.
2. `cat ~/Desktop/gvc-local/data/puzzles/tagged_connections.json > /dev/null`
   — the file is iCloud-evicted and unreadable through the bridge; this
   materializes it (needed for the C1.3 train-side cell).
3. Push the branch so the GPU box can clone it.

## 1. GPU box (Runpod A40 $0.44/h or L40S $0.99/h preferred; ~4–6 A40-hours total)

```bash
git clone https://github.com/jacksonmlukas/connections-rl && cd connections-rl
git checkout analysis/aug21
pip install -e ".[dev,train]" vllm && pip show trl vllm | grep -E "Name|Version" \
    | tee results-analysis/aug21/session-versions.txt
git clone --depth 1 https://github.com/jacksonmlukas/gvc-local /workspace/gvc-local
export CONNECTIONS_PUZZLES=/workspace/gvc-local/data/puzzles/tagged_connections.json
make data
huggingface-cli login   # your token
```

## 2. B1 training (~2–3 h on A40) — LAUNCH FIRST

```bash
python -m connections_rl.train.grpo --config results-analysis/aug21/taskB1_grpo-7b-noscale.yaml
```

The loud-fail guard aborts immediately if the installed TRL cannot honor
`scale_rewards` — that abort means upgrade TRL, never remove the guard.
Checkpoints sync to `connections-rl-grpo-7b-noscale-ckpt` every 50 steps.
**If it dies: one retry maximum, then report the death and move to step 3
with the checkpoints that exist.**

## 3. B1 checkpoint analyses (val split; ~1 h)

```bash
python -m connections_rl.eval.checkpoint_curve \
    --checkpoints artifacts/grpo-7b-noscale/checkpoint-*  \
    --puzzles data/splits/puzzles_val.json \
    --out results-analysis/aug21/ckpt-curve-7b-noscale   # flags per the module's argparse
python -m connections_rl.eval.entropy_kl \
    --model Qwen/Qwen2.5-7B-Instruct --load-in-4bit --sft-adapter artifacts/sft-7b \
    --checkpoints base=base sft=artifacts/sft-7b \
        ckpt-50=artifacts/grpo-7b-noscale/checkpoint-50 [... every 50 through the final step] \
    --puzzles data/splits/puzzles_val.json --n 100 --temperature 0.9 \
    --out results-analysis/aug21/entropy-kl-7b-noscale
```

Read the val-semantic argmax step off the curve; that is NOSCALE_PEAK_STEP.
Substitute it in `taskB2_eval_test.yaml` (the `noscale-peak` comment and the
`--lora-modules` line below).

## 4. One vLLM session for every eval + the C2 capture (~2 h)

```bash
huggingface-cli download jacksonlukas/connections-rl-grpo-7b-ckpt --local-dir adapters/grpo-7b-ckpt
huggingface-cli download jacksonlukas/connections-rl-grpo-7b     --local-dir adapters/grpo-7b
# original run step-100 comes from the same ckpt repo (checkpoint-100/)

vllm serve Qwen/Qwen2.5-7B-Instruct --dtype half --enable-lora --enforce-eager \
  --max-lora-rank 16 --max-loras 2 --max-model-len 2048 --gpu-memory-utilization 0.85 \
  --lora-modules \
    connections-rl-grpo-7b-ckpt50=adapters/grpo-7b-ckpt/checkpoint-50 \
    connections-rl-grpo-7b-ckpt100=adapters/grpo-7b-ckpt/checkpoint-100 \
    connections-rl-grpo-7b=adapters/grpo-7b \
    connections-rl-grpo-7b-noscale=artifacts/grpo-7b-noscale \
    connections-rl-grpo-7b-noscale-peak=artifacts/grpo-7b-noscale/checkpoint-NOSCALE_PEAK_STEP
# (drop --tensor-parallel-size on a single A40/L40S; keep -tp 2 only on 2xT4)

python results-analysis/aug21/c1_shift_analysis.py prepare
python -m connections_rl.eval.run --config results-analysis/aug21/taskB2_eval_test.yaml
python -m connections_rl.eval.run --config results-analysis/aug21/c11_eval_trainslice.yaml
python results-analysis/aug21/c2_capture_generations.py
python results-analysis/aug21/taskB2_paired.py
python results-analysis/aug21/c1_shift_analysis.py analyze
python results-analysis/aug21/r_map_deliverables.py   # exact data/ paths + R6 26/77/4 gate
```

(The R6 captures come free with the eval: `taskB2_eval_test.yaml` sets
`capture_generations: true`, an opt-in harness flag — default off everywhere
else — that writes `<arm>/generations.jsonl` per puzzle as
`{"puzzle_id","prompt","generation","groups_correct","valid"}`.)

## 5. C2 judging (CPU anywhere, needs OPENROUTER_API_KEY; a couple of dollars)

```bash
export OPENROUTER_API_KEY=...   # owner-side only
python results-analysis/aug21/c2_judge_openrouter.py
```

## 6. B3 + the 1.5B step count (local, your W&B login)

```bash
pip install wandb && python results-analysis/aug21/b3_wandb_export.py
```

Writes `data/wandb_train_reward.csv` (R1: raw `scan_history` rows, unsmoothed)
and `data/step_count_1p5b.json` (R2: the final logged `_step` settles
403-vs-806; if W&B lacks the run the script says so and writes nothing —
the number is never inferred from the config).

## 7. Persist

```bash
zip -qr aug21-outputs.zip results-analysis/aug21 data -x 'data/splits/*'
python - <<'PY'
from huggingface_hub import HfApi
HfApi().upload_folder(folder_path="results-analysis/aug21",
    repo_id="jacksonlukas/connections-rl-results", repo_type="dataset", path_in_repo="aug21")
HfApi().upload_folder(folder_path="data",
    repo_id="jacksonlukas/connections-rl-results", repo_type="dataset",
    path_in_repo="aug21/data", ignore_patterns=["splits/*"])
PY
```

Then commit `results-analysis/aug21/` on the branch and push. The agent
re-verifies everything from the records once the outputs land (Hub or disk).

## The four explicit answers this bundle must produce

1. **B1:** does the semantic hump still appear without group-std scaling, and
   does the final policy still land below base? (checkpoint curve + evalB)
2. **B2:** is step 100 on test close to step 50 (plateau) or already fallen
   (sharper collapse)? (evalB + paired)
3. **C1:** do any of the three shift measurements differ from zero? (two are
   already computed: the date regressions are null; C1.1 and the train mix
   complete in-session)
4. **C2:** does the blind judge rank step-50 > base > final, or does it prefer
   the collapsed policy's well-formed output?
