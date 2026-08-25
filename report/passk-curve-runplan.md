# Run plan: the full pass@k curve (k = 1, 2, 4, 8, 16, 32)

Target: all three 7B arms (base, SFT, GRPO seed 0) on the 162-puzzle test set,
k ∈ {1, 2, 4, 8, 16, 32}. Estimation code is committed and unit-tested
(`src/connections_rl/eval/passk_curve.py`); only the generation step needs a
GPU. This plan records every decision the writeup will have to defend.

## Decision: regenerate everything at n = 32

The existing k=16 pool (`results-analysis/passk-7b.json`) cannot be reused for
any point on the curve, for any metric. Its per-puzzle record schema is
`puzzle_id, date, any_solved, any_valid, max_groups_correct, best_reward,
n_samples` — aggregates over the 16 samples. `any_solved`/`any_valid` are
booleans (an OR), not success counts, so the `c` needed by the unbiased
binary estimator is unrecoverable; `max_groups_correct` is a maximum, so the
per-sample score multiset needed by the graded estimator is unrecoverable too
(defect D5). Neither the solve curve nor the groups curve is salvageable from
the existing pool below k = 16. Its `top_p` and vLLM version are also
unrecorded (D6). Per the standing rule against mixing pools with unverifiable settings,
the whole curve is computed from one fresh n = 32 pool per arm, generated in a
single serving session. The k=16 point of the new curve will be compared
against the old pool's best-of-16 values as a sanity check (they should agree
within sampling noise, not exactly).

## Estimator (stated for the caption)

- Whole-puzzle solve and validity: unbiased pass@k (Chen et al., Codex),
  per puzzle 1 − C(n−c, k)/C(n, k), n = 32, averaged over 162 puzzles,
  computed as a stable running product.
- Groups correct: exact expectation of the max over a uniformly random
  k-subset of the 32 per-sample scores (order-statistics closed form) ÷ 4,
  reported as percent of groups. At k = n this equals the observed max, i.e.
  the same best-of-k definition as the existing pass@16 numbers.
- No point is subsampled or extrapolated; k ≤ n holds for every reported k.

## Success predicate (identical across all k and arms)

`reward_breakdown` on the raw completion — the same scorer as the greedy
tables and the old pool: `solved` = all four groups correct; `groups_correct`
= count of fully-correct groups (0–4); `valid` = well-formed 4×4 partition of
the 16 board words.

## Sampling settings (held constant across arms, recorded in the pool JSON)

| Setting | Value |
|---|---|
| n samples/puzzle | 32 |
| temperature | 0.9 (matches the k=16 pool and GRPO training temperature) |
| top_p | 1.0, explicit |
| max_tokens | 512 |
| serving | one vLLM session, all three arms via `--lora-modules`, same layout as `configs/eval/qwen7b.yaml` |
| vLLM version | export `CRL_VLLM_VERSION=$(pip show vllm | grep Version)` before generating |

## Steps

```bash
# 0. serve (GPU box): vLLM with base + sft-7b + grpo-7b adapters, one session
# 1. generate the pool (~32/16 = 2x the old pool's token budget)
export CRL_VLLM_VERSION="$(pip show vllm 2>/dev/null | awk '/^Version/{print $2}')"
python -m connections_rl.eval.passk_curve generate \
    --config configs/eval/qwen7b.yaml --n 32 --temperature 0.9 --top-p 1.0 \
    --out results-analysis/passk-pool-7b-n32.json

# 2. compute the curve + plot (no GPU; deterministic from the pool)
python -m connections_rl.eval.passk_curve curve \
    --pool results-analysis/passk-pool-7b-n32.json --k 1 2 4 8 16 32 \
    --out results-analysis/passk-curve-7b.json \
    --plot results-analysis/passk-curve-7b.png
```

Commit the pool, the curve JSON, and the plot together so the figure is
reproducible from committed data.

## Acceptance checks before writing up

1. Every puzzle has exactly 32 samples in every arm (the script warns
   otherwise; a shortfall caps usable k and must be stated).
2. The new k=16 groups values agree with the old pool's best-of-16 within
   bootstrap noise; if they do not, report the disagreement, do not reconcile.
3. The curve's k=1 values will NOT match the greedy tables (temperature 0.9
   expectation vs temperature-0 argmax); say so in the caption rather than
   treating it as a bug.
4. State n = 32 in every caption; report whether the base and GRPO lines cross
   anywhere in range — whatever the data shows.
5. Fallback: if 32 samples/puzzle is too expensive for the 7B arms on the
   available GPU, run n = 16 and cap the curve at k = 16 with one plain
   sentence saying the k = 32 column was not computed.
