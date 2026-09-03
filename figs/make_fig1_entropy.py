#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Figure 1 -- the mechanism, two panels from the n=100 temperature-0.9 entropy
sweep (results-analysis/entropy-kl-7b.json; the reported 7B run, seed 0).

Left:  policy entropy vs optimizer step (the collapse).
Right: semantic score vs KL(policy || SFT init) per sequence (the
       over-optimization curve), informative steps labeled.

Built 2026-08-24 for the rebuilt tae_submission.tex. Regenerated from the
JSON artifact -- never reuse a rendered PNG whose provenance predates the
current numbers. Run from the repo root:
    python3 figs/make_fig1_entropy.py
Output: figs/fig1_entropy_collapse.{png,pdf}
"""
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42  # never Type 3
matplotlib.rcParams["ps.fonttype"] = 42

import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "results-analysis", "entropy-kl-7b.json")
OUT = os.path.join(ROOT, "figs")

pts = json.load(open(SRC))
for p in pts:
    if p["n_puzzles"] != 100:
        sys.exit("ERROR: %s row %r has n_puzzles=%s, expected 100 -- wrong sweep?"
                 % (SRC, p["name"], p["n_puzzles"]))
traj = sorted([p for p in pts if p["step"] >= 0], key=lambda p: p["step"])
base = next((p for p in pts if p["step"] < 0), None)
# guard against a stale artifact: the sweep must span the SFT init through 403
assert traj[0]["step"] == 0 and traj[-1]["step"] == 403, \
    "sweep does not span step 0..403 -- stale or wrong artifact"

fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.2), dpi=300)

ax = axes[0]
ax.plot([p["step"] for p in traj], [p["entropy_per_token"] for p in traj],
        "o-", ms=3.5, lw=1.4, color="#C1440E", label="GRPO trajectory")
if base is not None:
    ax.axhline(base["entropy_per_token"], ls="--", lw=1.0, color="#777777",
               label="base (untrained)")
ax.axvline(50, ls=":", lw=0.9, color="#444444")
ax.annotate("semantic peak\n(step 50)", xy=(50, 0.23), fontsize=6.4,
            ha="left", xytext=(62, 0.215))
ax.set_xlabel("GRPO optimizer step", fontsize=8)
ax.set_ylabel("Policy entropy (nats/token)", fontsize=8)
ax.tick_params(labelsize=7)
ax.legend(fontsize=6.4, frameon=False)
ax.spines[["top", "right"]].set_visible(False)

ax = axes[1]
kl = [p["kl_from_sft_per_sequence"] for p in traj]
sem = [p["semantic_groups_correct"] for p in traj]
ax.plot(kl, sem, "o-", ms=3.5, lw=1.4, color="#1F5F8B", label="GRPO trajectory")
seen = []
for p, x, y in zip(traj, kl, sem):
    if any(abs(x - sx) < 1.5 and abs(y - sy) < 0.006 for sx, sy in seen):
        continue  # post-collapse points pile up; label once
    seen.append((x, y))
    ax.annotate(str(p["step"]), (x, y), fontsize=6.2,
                xytext=(3, 3), textcoords="offset points")
if base is not None:
    ax.scatter([base["kl_from_sft_per_sequence"]], [base["semantic_groups_correct"]],
               marker="x", s=36, color="#777777", zorder=5, label="base (untrained)")
ax.set_xlabel("KL(policy $\\Vert$ SFT init), nats/sequence", fontsize=8)
ax.set_ylabel("Semantic score (groups/4)", fontsize=8)
ax.tick_params(labelsize=7)
ax.legend(fontsize=6.4, frameon=False, loc="upper right")
ax.spines[["top", "right"]].set_visible(False)

fig.tight_layout()
os.makedirs(OUT, exist_ok=True)
for ext in ("png", "pdf"):
    fig.savefig(os.path.join(OUT, "fig1_entropy_collapse." + ext), bbox_inches="tight")
print("wrote figs/fig1_entropy_collapse.png / .pdf from %s (%d trajectory points)"
      % (os.path.relpath(SRC, ROOT), len(traj)))
