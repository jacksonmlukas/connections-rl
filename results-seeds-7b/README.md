# `results-seeds-7b/` — 7B three-seed endpoint session

Held-out eval of the three *leaked* 7B GRPO seeds (`grpo-seed0/`, `grpo-seed1/`,
`grpo-seed2/`) against a shared `sft/` baseline, all decoded in one vLLM serving
session so the seeds are mutually comparable.

**Cited by:** the paper's Table 6 (7B endpoints across three seeds), which the
caption explicitly marks as a *separate serving session* from Table 2 — the SFT
arm here reads 52/648 at 0.2284 invalid, against the control session's 52/648 at
0.2222. The repo README's seed-replication table cites the same files.

**These files are an immutable record.** Regenerate elsewhere; never edit in place.
