# `results-7b/` — 7B main results (original era)

Offline eval of the three 7B arms (`base/`, `sft/`, `grpo/`) on the same
162-puzzle test split, plus `comparisons.json`. This is the *leaked* GRPO run
(seed 0, step 403); see the repo README for what that means.

**Cited by:** the repo README's historical 7B scale-ablation table and the
pass@1 column of its pass@16 table. Note this is **not** the source of the
paper's Table 2: base (26/648) and GRPO (4/648) agree, but the SFT row here is
0.3457 (56/648) versus the paper's 0.3210 (52/648). The paper's Table 2 is a
single-session measurement from `results-analysis/aug27/memC-session-test/`;
the gap is the documented cross-session decode spread (SFT decodes 52–56 slots
across the four observed sessions, per the paper's Table 6 caption).

**These files are an immutable record.** Regenerate elsewhere; never edit in place.
