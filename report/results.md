# Results

| Arm | n | Solve rate (95% CI) | Invalid rate | Groups correct (mean) | One-away rate |
|---|---|---|---|---|---|
| gvc-local basic (8B) (reference) | 10 | 20.0% [0.0, 50.0] | — | — | — |
| gvc-local GVC multi-agent (8B) (reference) | 10 | 60.0% [30.0, 90.0] | — | — | — |
| base | 162 | 0.0% [0.0, 0.0] | 32.1% [24.7, 38.9] | 0.01 | 0.0% [0.0, 0.0] |
| grpo | 162 | 0.0% [0.0, 0.0] | 2.5% [0.6, 5.6] | 0.01 | 0.0% [0.0, 0.0] |
| sft | 162 | 0.0% [0.0, 0.0] | 74.1% [67.3, 80.2] | 0.01 | 0.0% [0.0, 0.0] |


## Significance (paired, same puzzles)

```json
{
  "base_vs_sft": {
    "n_paired": 162,
    "a_solve_rate": 0.0,
    "b_solve_rate": 0.0,
    "mcnemar_p": 1.0,
    "solve_rate_diff_ci": [
      0.0,
      0.0,
      0.0
    ],
    "discordant_a_only": 0,
    "discordant_b_only": 0
  },
  "base_vs_grpo": {
    "n_paired": 162,
    "a_solve_rate": 0.0,
    "b_solve_rate": 0.0,
    "mcnemar_p": 1.0,
    "solve_rate_diff_ci": [
      0.0,
      0.0,
      0.0
    ],
    "discordant_a_only": 0,
    "discordant_b_only": 0
  },
  "sft_vs_grpo": {
    "n_paired": 162,
    "a_solve_rate": 0.0,
    "b_solve_rate": 0.0,
    "mcnemar_p": 1.0,
    "solve_rate_diff_ci": [
      0.0,
      0.0,
      0.0
    ],
    "discordant_a_only": 0,
    "discordant_b_only": 0
  }
}
```