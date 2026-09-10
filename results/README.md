# `results/` — 1.5B main results (original era)

Offline eval of the three 1.5B arms (`base/`, `sft/`, `grpo/`) on the 162-puzzle
chronological test split, plus `comparisons.json` (paired McNemar/bootstrap
between arms) and `smoke/` (the tiny bundled-fixture run that `make eval-smoke`
and CI reproduce).

**Cited by:** the paper's Table 8 (1.5B test split) reads directly off
`{base,sft,grpo}/metrics.json` — base 0.0062 groups / 32.1% invalid / 0.049
reward, SFT 0.0123 / 74.1% / −0.038, GRPO 0.0062 / 2.5% / 0.113. The repo
README's historical 1.5B table cites the same three files.

**These files are an immutable record.** They are the exact bytes the published
numbers were read from. Regenerate into a new directory; never edit in place.
Note that `make eval-smoke` rewrites `smoke/` — revert it if you run it.
