# aug20 analysis pass — index and provenance

Produced 2026-08-20 against the committed artifacts; nothing outside this
directory was created or modified. Every value in these files carries its
scale, session, and source path. Sessions: A = `results/`, `results-7b/`;
B = `results-seeds-*/`.

| File | Task | Status |
|---|---|---|
| `taskA_1p5b_groups.json` | A — 1.5B groups-correct + solve, both sessions | done (CPU, from committed artifacts) |
| `taskB_conditional_on_valid.json` | B — groups-correct over valid outputs only, both scales | done (exact, from records.jsonl) |
| `taskC_7b_per_category.json` | C — 7B per-category stratification + base→GRPO deltas | done (metrics.json cross-checked against records) |
| `taskE_stratified_bootstrap.json` | E — joint puzzle×seed bootstrap vs puzzles-only | done (10,000 resamples, two independent RNG streams) |
| `taskD-session/` (+ `taskD_eval_ckpt50.yaml`, `taskD_paired_groups.py`, `TASKD_RUNBOOK.md`, `taskD_kaggle.ipynb`) | D — step-50 on test | **done** — run 2026-08-20/21 on Kaggle (vLLM 0.27.1, one session, base + ckpt-50 + final); mirrored from `hf://datasets/jacksonlukas/connections-rl-results/aug20/` with byte-exact and full-precision metric verification. Headline: ckpt-50 groups 0.4753 [0.364, 0.599] (0-4) vs base 0.1605; paired ckpt50−base +0.3148 [+0.204, +0.438], final−base −0.1358 [−0.191, −0.080], seed-0 bootstrap independently reproduced to full precision; both anchor arms reproduce session A per-puzzle exactly (0/162 differences) |
| `taskF_grpo-7b-beta01.yaml` + runbook §F | F — β=0.01 run | prepared; only after D |

Headline readouts (details and exact values in the JSONs):

- **A.** 1.5B groups-correct (0-4, session A): base 0.00617, SFT 0.01235,
  GRPO 0.00617 — GRPO equals base exactly; solve 0.0% for every arm in both
  sessions. The scale-contrast wording must be "nothing to lose at 1.5B".
- **B.** Conditioning on valid output does not rescue 7B GRPO: valid-only
  means (session A) base 0.1722 (n=151), SFT 0.4444 (n=126), GRPO 0.0248
  (n=161). The conditional gap is larger than the unconditional one, so the
  headline deficit is not a validity artifact.
- **C.** Base→GRPO relative loss by category (session A): category −92%,
  tag-fillin −86%, cultural −67%, wordplay −100% (base was already near floor
  at 0.048, n=21), silent-letter 0→0 (n=4). The profile is closer to uniform
  destruction than to the predicted wordplay-worst / category-least gradient;
  reported as evidence either way per the brief.
- **E.** SFT−GRPO (0-4, session B, mean over seeds): +0.2757; puzzles-only CI
  [0.1687, 0.3889]; joint puzzle×seed CI [0.1687, 0.3909]. Between-seed sd
  0.0177 is small relative to puzzle noise, so the corrected interval barely
  widens and no conclusion changes.
- **Open Q1 (KL convention).** Lives in `src/connections_rl/eval/entropy_kl.py`
  (not `checkpoint_curve.py`): sample-based E_{y~π}[log π − log ref], i.e.
  KL(policy ‖ ref), with `kl_from_sft_per_sequence` the per-sequence form —
  47.05 (final GRPO) and 42.60 (base) are the same convention and units.
  Caveat for the prose: each expectation is under its own policy's samples
  (9,766 base vs 7,928 GRPO tokens over the same 100 puzzles).
- **Open Q2 (human/agreement statistic).** Grep across all *.py, *.md,
  *.json, *.yaml, *.ipynb for inter-rater/kappa/krippendorff/annotator/human
  baseline/agreement terms: zero hits. No such statistic exists in the
  artifacts; the "no human baseline" concession is safe to make final.
