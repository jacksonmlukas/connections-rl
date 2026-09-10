# `results-analysis/` — dated analysis sessions

One subdirectory per analysis session, named by the date it was run. Each holds
the configs, notebooks, serving-session eval outputs, and summary JSON for that
session. Top-level files are cross-session artifacts: `entropy-kl-7b.json`
(per-checkpoint policy entropy and KL from the RL init), `ckpt-curve-7b.json`
(the leaked run's validation grid), `passk-7b.json` and `passk-1.5b.json`
(best-of-16 sampling budget), plus their `.png` renders.

| Session | What it produced |
|---|---|
| `aug20/` | First post-hoc pass over the committed artifacts: per-category stratification, conditional-on-valid scores, paired tests. Has its own README. |
| `aug21/` | Post-review experiments: the `scale_rewards` normalization ablation (`evalB-session/`, `ckpt-curve-7b-noscale.json`), step-100 eval, and the W&B exports that produced `data/wandb_train_reward.csv`. Has its own README. |
| `aug27/` | **The diagnosis.** `GRPO_PROMPT_FIX.md` describes the data-loader leak; `memC-session-{train,test}/` is the four-arm control session; `copy_rule_results.json` scores the positional copy rule offline. |
| `aug29/` | First leak-free rerun (seed 0) with the puzzle-seeded shuffle: `leakfree-session-test/`, the checkpoint curve, and the full 2.4 MB training log. |
| `sep04/` | `answer_ordered_kaggle.ipynb` — answer-ordered eval, **queued but not yet run**; no outputs. |
| `sep05/` | Leak-free seeds 1–2 plus the leaked-SFT arm: `finals-session-test/`, `peaks-session-test/`, per-seed checkpoint curves. |
| `sep06/` | `empirical_floor.json` — how often a positional copy rule actually wins on shuffled boards (3 of 807 train, 1 of 162 test). |
| `sep07/` | Step-50 replication on the leaked seeds: `step50-session-test/`. |

**Cited by:** the paper's Table 2 and Table 3 (`aug27/memC-session-*`), Table 4
(`aug29/leakfree-session-test/`), Table 5 and Figure 1 (`entropy-kl-7b.json`),
Table 7 (`aug27/` copy-rule files), Appendix E (`aug29/leakfree-kaggle-run.log`),
and the normalization ablation of §6 (`aug21/evalB-session/`).

**These files are an immutable record.** They are the exact bytes the published
numbers were read from — including the large logs and `generations.jsonl` files,
whose size is the point. Regenerate into a new dated directory; never edit in place.
