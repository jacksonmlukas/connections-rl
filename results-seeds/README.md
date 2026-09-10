# `results-seeds/` — cross-seed weight-space analysis

Not an eval session. Three files summarising how the *parameters* moved across
GRPO seeds: `seed_summary.json` (per-seed endpoint metrics, both scales) and
`weight_space_{1.5b,7b}.txt` (cosine similarity between seeds' RL-induced LoRA
updates, per-module magnitude comparisons). Produced by
`scripts/analyze_seed_weightspace.py`.

**Cited by:** the repo README's "Weight-space convergence" paragraph (+0.67 to
+0.69 cosine at 7B, +0.78 to +0.80 at 1.5B).

**These files are an immutable record.** Regenerate elsewhere; never edit in place.
