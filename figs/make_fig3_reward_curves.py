#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Figure 3 -- training reward and held-out reward, two stacked panels, shared step axis.

Rebuilt 2026-08-24 from the recovered original, with three fixes against the
real W&B export (data/wandb_train_reward.csv, raw scan_history):
  1. The step column is train/global_step (the optimizer step). W&B's _step is
     a log-call counter (0..11364 for this run) and must never be the x-axis.
  2. The reward column is train/reward exactly; substring matching would hit
     the profiling/..._calculate_rewards timing column first.
  3. The held-out anchors are the aug27 memC control session (the paper's
     Tables 2/3 session; step-50 anchor 0.2519 there vs 0.2463 in aug20 Task D);
     step 100 exists only in the aug21 evalB session and is drawn as a hollow
     marker labeled as such -- sessions are never mixed on one curve. The
     held-out series is drawn on a twin right axis so its 0.125..0.266 range
     is legible against the 0.99..1.60 training curve.

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
# aug27 memC control session (the four-arm session of the paper's Tables 2/3;
# artifact: results-analysis/aug27/memC-session-test/*/metrics.json). Base and
# final reproduce the aug20 Task D session per-puzzle exactly; step 50 differs
# across sessions (0.2463 in aug20 Task D), which is why the session is named:
HELDOUT = [(50, 0.2518518518518518), (403, 0.125)]
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

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6.4, 2.05), dpi=300, sharex=True,
                                   gridspec_kw=dict(height_ratios=[0.85, 1], hspace=0.13))
    # Top panel: the in-sample story on its own scale. No twin axes anywhere --
    # the two measurements of the same reward get one honest scale each, and the
    # vertical stack makes the contrast readable at a glance.
    ax1.plot([s for s, _ in train], [r for _, r in train], "-", lw=1.5, color="#909090")
    ax1.axhline(REWARD_CEILING, ls="--", lw=1.0, color="#888888")
    ax1.annotate("ceiling %.1f, reached step %d" % (REWARD_CEILING, CEILING_STEP),
                 xy=(400, REWARD_CEILING), xytext=(0, -3), textcoords="offset points",
                 ha="right", va="top", fontsize=8.2, color="#666666")
    ax1.set_ylim(0.0, 1.82)
    ax1.set_yticks([0, 0.8, 1.6])
    ax1.set_ylabel("Training\nreward", fontsize=9.5)
    ax1.tick_params(labelsize=9)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    # Bottom panel: held out. The segment between the two anchors is UNMEASURED
    # (no held-out eval exists between steps 50 and 403 in this session): drawn
    # dashed and faded, labeled, never solid.
    ax2.plot([s for s, _ in HELDOUT], [r for _, r in HELDOUT], "--", lw=1.3,
             dashes=(4, 3), color="#C1440E", alpha=0.5, zorder=2)
    ax2.plot([s for s, _ in HELDOUT], [r for _, r in HELDOUT], "o", ms=6,
             color="#C1440E", linestyle="none", zorder=4)
    ax2.plot([HELDOUT_B2[0]], [HELDOUT_B2[1]], "o", ms=6, mfc="none", mec="#C1440E",
             mew=1.4, zorder=4)
    ax2.axhline(BASE_REWARD, ls=":", lw=1.1, color="#444444", zorder=1)
    ax2.annotate("untrained Instruct, %.3f" % BASE_REWARD, xy=(8, BASE_REWARD),
                 xytext=(0, 2), textcoords="offset points", ha="left", va="bottom",
                 fontsize=8.2, color="#444444")
    ax2.annotate("%.3f" % HELDOUT[0][1], xy=(50, HELDOUT[0][1]), xytext=(-8, 0),
                 textcoords="offset points", ha="right", va="center",
                 fontsize=8.4, color="#C1440E")
    ax2.annotate("%.3f  (later session)" % HELDOUT_B2[1], xy=HELDOUT_B2, xytext=(8, 0),
                 textcoords="offset points", ha="left", va="center",
                 fontsize=8.0, color="#C1440E")
    ax2.annotate("%.3f" % HELDOUT[-1][1], xy=HELDOUT[-1], xytext=(-2, 9),
                 textcoords="offset points", ha="right", va="bottom",
                 fontsize=8.4, color="#C1440E")
    ax2.annotate("dashed span unmeasured", xy=(150, 0.194), ha="left", va="top",
                 fontsize=7.6, color="#8a3006", style="italic")
    ax2.set_ylim(0.095, 0.305)
    ax2.set_yticks([0.10, BASE_REWARD, 0.25])
    ax2.set_yticklabels(["0.10", "0.165", "0.25"])
    ax2.set_ylabel("Held-out\nreward", fontsize=9.5)
    ax2.tick_params(labelsize=9)
    ax2.set_xlabel("GRPO optimizer step  (Qwen2.5-7B-Instruct)", fontsize=9.5)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    # no in-figure title: the LaTeX caption carries it

    os.makedirs(OUT, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(OUT, "fig3_reward_curves." + ext), bbox_inches="tight")
    print("wrote figs/fig3_reward_curves.png / .pdf  (columns: %r, %r)" % (step_key, rew_key))


if __name__ == "__main__":
    main()
