# aug21 — post-review experiments: index and status

Branch `analysis/aug21`. Prepared 2026-08-20 (US/Eastern evening); GPU tasks are owner-launched via `aug21_kaggle.ipynb` (Kaggle T4 x2, import-and-run;
resumes B1 from the Hub ckpt repo if the 12 h cap hits). `RUNBOOK_aug21.md` holds
the same sequence in shell form for any other box. Every computed number carries
scale, session, and source path.

| Item | File(s) | Status |
|---|---|---|
| B1 scale_rewards=False run | `taskB1_grpo-7b-noscale.yaml`, `taskB1_grpo_scale_rewards.patch` | prepared — **requires the grpo.py patch first** (config alone is silently ignored; the patch fails loudly on old TRL) |
| B2 step-100 on test | `taskB2_eval_test.yaml`, `taskB2_paired.py` | prepared — rides the single aug21 eval session with base/step-50/final/B1 arms |
| B3 training-reward series | `b3_wandb_export.py` | prepared — owner runs locally with W&B login; also settles the 1.5B 403-vs-806 question |
| C1.1 base on train slice | `c11_train_sample.json`, `c1_shift_analysis.py`, `c11_eval_trainslice.yaml` | sample **prespecified** (random.Random(0) over canonical train ids; split replication validated: 807/108/162, test = ids 917–1078; loader drops id 660, 2025-04-01, "only 1 unique words") — eval rides the same session |
| C1.2 date regression | `c12_date_regression.json` | **done** — base −0.158 groups/yr [−0.634, +0.292], final GRPO +0.035 [−0.271, +0.329] (0-4 scale, session D records, 10k bootstrap, seed 0): both indistinguishable from zero |
| C1.3 stratum mix | `c13_stratum_mix.json` | **partial** — test side done (category .426 / tag-fillin .247 / cultural .173 / wordplay .130 / silent-letter .025); train side pending `tagged_connections.json` (iCloud-evicted, unreadable via bridge — open it once locally), computed by `c1_shift_analysis.py analyze` in-session |
| C2 LLM-judge row | `c2_capture_generations.py`, `c2_judge_openrouter.py` | prepared — **generations are not stored anywhere in the repo** (records keep scores only), so the capture step in the eval session is mandatory before judging |

Key preparation findings (all artifact-backed):

- `train/grpo.py` builds `GRPOConfig` via `compatible_config_kwargs` over a
  fixed dict with no `scale_rewards` key — the B1 ablation is impossible from
  config alone, and a version-filtered kwarg would silently replicate the
  original run; hence the patch's loud-fail guard.
- The split replicates exactly from the raw DB: sort by (date, id), val_frac
  0.10 / test_frac 0.15 with tie push-back → 807/108/162 after the loader
  drops puzzle 660 (the 2025-04-01 board has one unique word). Test = ids
  917–1078, matching every published table.
- Raw generations were never persisted by the eval harness; C2 requires the
  capture pass.
