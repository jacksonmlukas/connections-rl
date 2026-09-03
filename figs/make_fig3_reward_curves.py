#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Figure 3 -- training reward and held-out reward on one axis against training step.

Rebuilt 2026-08-24 from the recovered original, with three fixes against the
real W&B export (data/wandb_train_reward.csv, raw scan_history):
  1. The step column is train/global_step (the optimizer step). W&B's _step is
     a log-call counter (0..11364 for this run) and must never be the x-axis.
  2. The reward column is train/reward exactly; substring matching would hit
     the profiling/..._calculate_rewards timing column first.
  3. The held-out anchors are the aug20 Task D session (base/step-50/final);
     step 100 exists only in the aug21 evalB session and is drawn as a hollow
     marker labeled as such -- sessions are never mixed on one curve.

Run from the repo root:  python3 figs/make_fig3_reward_curves.py
Input:  data/wandb_train_reward.csv   Output: figs/fig3_reward_curves.{png,pdf}
"""
import csv
import os
import sys

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42  # never Type 3: arXiv flags bitmap fonts
matplotlib.rcParams["ps.fonttype"] = 42
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
OUT = os.path.join(ROOT, "figs")

REWARD_CEILING = 1.6  # RewardConfig.max_reward = 0.1 + 1.0 + 0.5
CEILING_STEP = 125    # first logged step at (1.6000, std 0.0000); memo aug24

# ---- held-out (test, 162 puzzles, greedy) mean reward ----------------------
# aug20 Task D session (single vLLM session; base and final reproduce other
# sessions per-puzzle exactly):
HELDOUT = [(50, 0.2462962962962963), (403, 0.125)]
BASE_REWARD = 0.16543209876543210  # untrained base, same session
# aug21 evalB session (a DIFFERENT serving session; step-50 re-serves there at
# 0.2525). Drawn as a hollow marker, never joined to the Task D curve:
HELDOUT_B2 = (100, 0.266358024691358)

# Consistency notices, NOT asserts: the figure renders whatever the constants
# say. A plotting script that refuses to draw a null result would bake the
# paper's conclusion into its own evidence.
if not (HELDOUT[0][1] == max(r for _, r in HELDOUT) and HELDOUT[0][1] > BASE_REWARD):
    print("NOTICE: step 50 is not the held-out peak / does not exceed base in these "
          "constants. Rendering anyway -- update the paper's prose, not this script.")
if not HELDOUT[-1][1] < BASE_REWARD:
    print("NOTICE: the final policy does not end below base in these constants. "
          "Rendering anyway -- update the paper's prose, not this script.")


def load_training_curve():
    csvp = os.path.join(DATA, "wandb_train_reward.csv")
    if not os.path.exists(csvp):
        sys.exit("ERROR: %s not found. It is the R1 deliverable (also on the Hub at "
                 "aug21/data/wandb_train_reward.csv)." % csvp)
    rows = list(csv.DictReader(open(csvp)))
    if not rows:
        sys.exit("ERROR: %s is empty." % csvp)
    cols = list(rows[0])
    # Optimizer step: train/global_step (or bare global_step). NEVER _step.
    step_key = next((k for k in cols if k.strip().lower().endswith("global_step")), None)
    if step_key is None:
        sys.exit("ERROR: no train/global_step column in %s. Refusing to fall back to "
                 "_step, which is a log-call counter, not the optimizer step.\n"
                 "Columns present: %s" % (csvp, cols))
    # Reward: exact train/reward first; profiling columns also contain 'reward'.
    for cand in ("train/reward", "reward"):
        rew_key = next((k for k in cols if k.strip().lower() == cand), None)
        if rew_key:
            break
    if rew_key is None:
        sys.exit("ERROR: no train/reward column in %s.\nColumns present: %s" % (csvp, cols))
    print("  using columns: step=%r reward=%r" % (step_key, rew_key))
    pts = []
    for r in rows:
        try:
            pts.append((float(r[step_key]), float(r[rew_key])))
        except (ValueError, TypeError):
            continue  # W&B pads unlogged rows with blanks
    if not pts:
        sys.exit("ERROR: no (step, reward) pairs after skipping blank rows.")
    return sorted(pts), step_key, rew_key


def main():
    train, step_key, rew_key = load_training_curve()
    steps = [s for s, _ in train]
    print("  loaded %d training points, steps %g..%g" % (len(train), min(steps), max(steps)))
    if max(steps) > 500:
        sys.exit("ERROR: max step %g -- this looks like the _step log counter, not "
                 "the 403-step optimizer axis. Wrong column." % max(steps))

    fig, ax = plt.subplots(figsize=(6.4, 2.6), dpi=300)
    ax.plot([s for s, _ in train], [r for _, r in train], "-", lw=1.4, color="#B0B0B0",
            label="Training reward (in sample)", zorder=2)
    ax.axhline(REWARD_CEILING, ls="--", lw=1.0, color="#888888", zorder=1)
    ax.annotate("reward ceiling, %.1f (reached step %d)"
                % (REWARD_CEILING, CEILING_STEP),
                xy=(max(steps) * 0.985, REWARD_CEILING), xytext=(0, -9),
                textcoords="offset points", ha="right", va="top",
                fontsize=6.8, color="#666666")
    ax.plot([s for s, _ in HELDOUT], [r for _, r in HELDOUT], "-o", lw=1.8, ms=5.5,
            color="#C1440E", label="Held-out reward (162 test puzzles, single serving session)",
            zorder=4)
    ax.plot([HELDOUT_B2[0]], [HELDOUT_B2[1]], "o", ms=5.5, mfc="none", mec="#C1440E",
            mew=1.4, label="Held-out, step 100 (later session; not joined)", zorder=4)
    ax.axhline(BASE_REWARD, ls=":", lw=1.1, color="#444444", zorder=1)
    ax.annotate("untrained Instruct, %.3f" % BASE_REWARD, xy=(max(steps) * 0.985, BASE_REWARD),
                xytext=(0, 4), textcoords="offset points", ha="right", va="bottom",
                fontsize=6.8, color="#444444")
    ax.annotate("held-out peak\nstep 50, %.3f" % HELDOUT[0][1], xy=HELDOUT[0],
                xytext=(78, 0.42), fontsize=6.8,
                arrowprops=dict(arrowstyle="->", lw=0.8, color="k", alpha=0.8))
    ax.annotate("ends BELOW Instruct\nstep 403, %.3f" % HELDOUT[-1][1], xy=HELDOUT[-1],
                xytext=(255, 0.30), fontsize=6.8,
                arrowprops=dict(arrowstyle="->", lw=0.8, color="k", alpha=0.8))
    ax.set_xlabel("GRPO optimizer step  (Qwen2.5-7B-Instruct)")
    ax.set_ylabel("Mean reward")
    ax.set_title("The same reward function, in sample and held out", loc="left", pad=6)
    ax.legend(loc="center right", frameon=False, fontsize=7.0)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    os.makedirs(OUT, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(OUT, "fig3_reward_curves." + ext), bbox_inches="tight")
    print("wrote figs/fig3_reward_curves.png / .pdf  (columns: %r, %r)" % (step_key, rew_key))


if __name__ == "__main__":
    main()
