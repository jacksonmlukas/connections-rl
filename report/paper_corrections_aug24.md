# Sentence-level corrections for the preprint — 2026-08-24

Three required corrections, one conditional, and two ready-to-paste additions.
Every number carries its count and source artifact; session labels are part of
the sentence wherever a comparison could otherwise be read cross-session.

**Two standing rules for every sentence in the paper (v2 of this memo,
after review):**

1. **Name the sweep in every validation-score sentence.** There are two
   validation measurements and they never share a label: the **greedy n=108
   checkpoint sweep** (`checkpoint_curve`; denominators of 432) and the
   **n=100 temperature-0.9 entropy sweep** (`entropy_kl`; denominators of
   400). Trap to avoid: noscale step-50 on the n=100 sweep is 35/400 —
   numerically identical to the ORIGINAL run's step-100 on that same sweep,
   and the numeral 35 also appears as noscale step-100 on the n=108 sweep
   (35/432). Same digits, three different facts.
2. **Name the serving session in every step-50 sentence.** The step-50
   checkpoint has two test counts: 77/648 (aug20 Task D — the published
   headline) and 80/648 (aug21 evalB — the session the step-100 comparison
   ran in). The step-100 comparison uses the aug21 pair (80 vs 76); the
   headline stays aug20 (77). Never put 77 and 76 in one sentence — the
   4-group within-session difference would silently read as 1.

---

## Correction 1 — training-reward saturation is at step 125, at the ceiling (R1)

**Locate:** the sentence claiming the training reward "saturates by roughly
step 270" (and any echo of "~270" elsewhere).

**Replace with:**

> The training reward saturates at the reward ceiling by step 125: the mean
> reward over the K=8 rollouts reaches 1.6000 with standard deviation 0.0000
> at step 125 — the maximum the reward function can emit (0.1 format + 1.0
> grouping + 0.5 solve bonus) — and averages 1.5814 over the 57 logged points
> from step 125 through 403, sitting exactly at the ceiling in 38 of them.

Source: `data/wandb_train_reward.csv` (raw run history of W&B run
`odyc3xnk`, logged every 5 steps, 81 reward points); ceiling from
`RewardConfig.max_reward` in `src/connections_rl/reward/reward.py`.
Trajectory: 0.2125 (step 5) → 0.5519 (50) → 0.9931 (100) → 1.5206 (120) →
1.6000 (125). Step 270 is unremarkable: steps 255–285 read 1.6, 1.6, 1.6,
1.5888, 1.6, 1.6, 1.6.

**Strongly recommended companion sentence (mechanism section):**

> TRL's own `frac_reward_zero_std` diagnostic — the fraction of prompt groups
> whose eight sampled rewards are identical, i.e. whose group-relative
> advantage is exactly zero — rises from 0.0 at step 50 to 1.0 at step 125
> and averages 0.946 over the remainder of training: from step 125 onward the
> optimizer receives no learning signal on nearly every prompt.

Same source CSV, column `train/frac_reward_zero_std`. **Describe this; do not
explain past it.** What drives the remaining 278 steps of movement (entropy
falling to ~0.011, KL to ~46 nats/seq) while ~95% of prompts carry zero
advantage is an open question — plausibly the ~5% of prompts that still carry
signal, compounded, but that is unmeasured. Report the diagnostic and the
trajectory side by side without asserting the causal chain between them.

**Optional abstract strengthening** (only if the abstract currently reads
"the training reward does not record the reversal"):

> The training reward not only fails to record the reversal — it reads
> perfect: from step 125 the policy earns the maximum possible reward on
> training prompts while its held-out semantic score collapses to below the
> untrained base model.

**Do NOT write** (unverified inference, keep as an open question or a
limitations sentence at most): that the SFT stage memorized the train-split
answers and GRPO merely sharpens onto them. The pieces are consistent (SFT
trained on the 807 canonical train answers; ceiling reward on train prompts;
near-zero val semantics) but neither SFT nor final GRPO has been evaluated
offline on train puzzles — C1.1 evaluated base only (24/648).

---

## Correction 2 — the 1.5B run is 806 steps, two epochs (R2)

**Locate:** `implementation_notes.md`'s last v2 row labeled "403 (final)",
and the two `TODO(Jackson)` markers holding the 1.5B step count in the tex.

**Fixes:**

- Label: "403 (final)" → **"403 (epoch 1 of 2)"**; the final row is **806**.
- TODO replacement sentence:

> The 1.5B run trained for 806 optimizer steps — two epochs over the 807
> training puzzles (W&B run `8ynmbuda`, final `train/global_step` = 806,
> `train/epoch` = 2) — versus 403 steps, one epoch, at 7B.

- Scale-comparison caveat, wherever 1.5B and 7B are compared directly:

> The scales are epoch-asymmetric: the 1.5B policy saw each training puzzle
> twice, the 7B policy once.

Source: `data/step_count_1p5b.json` (verdict field records the run id, the
rejection of the two same-named dead starts, and why W&B's `_step` — a
log-call counter reading 22729 — was not used). Anywhere the paper currently
writes 403 for 1.5B, it becomes 806.

---

## Correction 3 — the step-50 peak is a plateau; collapse is between 100 and 150 (R4/B2)

**Locate:** any sentence implying the collapse begins immediately after step
50, or treating step 50 as a knife-edge argmax of the checkpoint grid.

**Replace with:**

> Step 100 was additionally evaluated on the test split in a single serving
> session alongside step 50: 76/648 groups recovered (0.4691 [0.3519,
> 0.6049]) versus step 50's 80/648 (0.4938 [0.3827, 0.6235]) in the same
> session, paired difference −0.0247 [−0.1481, +0.0927] (n=162, seed-0
> bootstrap) — statistically indistinguishable, with step 100 recording three
> full solves to step 50's two. The peak is a plateau spanning at least steps
> 50–100; the collapse occurs between steps 100 and 150, where validation
> semantic score falls from 35/400 to 3/400.

Sources: `data/b2_step100_test.json` (aug21 evalB session; includes both
paired bootstraps), `results-analysis/entropy-kl-7b.json` (n=100 validation
series) for the 35→3 collapse window. Session note: these step-50 numbers
are the aug21 session's; the paper's published step-50 headline (77/648,
0.4753 [0.3642, 0.5989]) is the aug20 Task D session — cite one session per
comparison, never mixed.

---

## Conditional correction 4 — greedy-serving reproducibility, if claimed

If the paper anywhere claims greedy (temperature-0) evaluation is exactly
reproducible, qualify it:

> Base and final-checkpoint arms reproduced per-puzzle exactly across three
> independent vLLM sessions (0/162 differences). The step-50 checkpoint did
> not: it differed on 2/162 puzzles across sessions (77 vs 80 of 648 groups
> recovered) and on 6/162 completions between two greedy passes within one
> session — mid-training policies sit near argmax ties that batched inference
> resolves inconsistently.

Sources: aug20 `taskD-session/` vs aug21 `evalB-session/` records (per-puzzle
diff on ids 917, 949); `c2_generations.json` vs
`evalB-session/grpo-ckpt50/generations.jsonl` (6/162 text diffs).

---

## Ready-to-paste addition A — the scale_rewards ablation (kills the named confound)

> To test whether the collapse is an artifact of group-standardized
> advantages (`scale_rewards="group"`, the TRL default — verified as the
> default in trl 1.10.0, whose `GRPOConfig.__post_init__` normalizes
> `False` to `"none"`), we repeated the 7B run identically except with
> reward scaling disabled. The signature is unchanged on the greedy n=108
> checkpoint sweep: validation semantic score rises to 62/432 at step 50,
> falls to 35/432 by step 100 and 4/432 by
> step 150; on test, the final policy recovers 7/648 groups (0.0432 [0.0123,
> 0.0804]) — below base by −0.1173 [−0.1790, −0.0556] paired (n=162, seed 0)
> — while its validation-selected peak (step 50, frozen before any test
> puzzle was scored) recovers 87/648 (0.5370 [0.4134, 0.6668]). The hump,
> the collapse, and the below-base endpoint all survive; the phenomenon is
> not a library default.

Sources: `results-analysis/aug21/ckpt-curve-7b-noscale.json`,
`data/b1_noscale_test_metrics.json`, `data/b1_noscale_peak_test.json`,
`evalB-session/paired_groups.json`, `session-trl-version.txt`.

**Companion sentence — the entropy/KL trajectory also replicates:**

> The dynamics match as well: without reward scaling, policy entropy falls
> 0.2111 (step 50) → 0.1346 (100) → 0.0128 (150) and stays near 0.011
> through step 403 (the original run's endpoint: 0.0099 nats/token), while
> KL from the SFT init rises 2.75 → 17.84 → 45.64 nats per sequence and
> plateaus near 46 (original: 47.05); on the n=100 temperature-0.9 entropy
> sweep — a different measurement from the greedy n=108 checkpoint sweep
> above — semantic score reads 35/400 at step 50, 17/400 at 100, then 2/400
> from step 150 onward as structural validity climbs from 0.70 to 0.96.

Source: `data/b1_noscale_ckpt_curve.json` (n_puzzles=100 in every row;
mirrored at `aug21/entropy-kl-7b-noscale.json`). Internal control: the base
row reproduces the original series' base sample exactly (42.5952008 nats/seq
over the same 9,766 tokens — `entropy_kl` seeds per puzzle id, so both runs
sampled identical base generations).

## Ready-to-paste addition B — the blind-judge row (JUDGe)

> Three LLM judges (gpt-5-mini, gemini-2.5-flash, claude-sonnet-4-5, via
> OpenRouter) rated all 486 test generations blind — board words and
> completion only, no answer key — on a 1–10 scale (162 per arm per judge;
> 1458/1458 responses parsed). All three rank step-50 > base > final, and all
> nine pairwise per-puzzle differences exclude zero under a paired bootstrap
> (seed 0): step-50 − final is +0.599 [+0.438, +0.784], +3.253 [+2.815,
> +3.673], and +1.617 [+1.333, +1.895] respectively. Every judge rates the
> collapsed policy below the untrained base despite its near-perfect output
> format (1/162 invalid): the reward cannot record the reversal, but a blind
> judge can.

Sources: `results-analysis/aug21/c2_judge_results.json` (means re-derived
exactly from raw scores), `c2_judge_paired.json`. Truth on the judged texts:
27, 79, 4 of 648 (scored on the judged text itself — see the drift note in
Correction 4). Judge-score/truth correlation is positive everywhere
(0.274–0.706), lowest on the final arm where truth is nearly constant (range
restriction).
