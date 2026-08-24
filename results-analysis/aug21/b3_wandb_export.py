"""R1 + R2 (aug21): W&B exports. Run LOCALLY with your W&B login — the agent
holds no credentials.

  pip install wandb && python results-analysis/aug21/b3_wandb_export.py

R1 -> data/wandb_train_reward.csv   (reported 7B run; RAW logged rows via
      scan_history — no sampling, no smoothing, no truncation; keeps `_step`
      plus every key containing "reward")
R2 -> data/step_count_1p5b.json     ({"final_step","epochs_logged",
      "wandb_run_url","verdict"} — read off W&B, never inferred from config)

If a run cannot be found, this script says so plainly and writes nothing for
that item. Multiple name matches abort rather than pick silently.
Run ids are pinned in RUN_IDS (resolved 2026-08-24; see comment there).
"""

import csv
import json
import sys
from pathlib import Path

import wandb

# Resolved 2026-08-24 from the all-projects enumeration: the trainer sets
# run_name but no W&B project, so everything lives in project 'huggingface'.
# The 1.5B v2 name matched THREE runs; the two others are dead starts (state
# failed, _step 1). Explicit ids, chosen for that recorded reason:
RUN_IDS = {
    "7b": "odyc3xnk",     # connections-rl-grpo-qwen7b-v1, finished
    "1.5b": "8ynmbuda",   # connections-rl-grpo-qwen1.5b-v2, finished
}
PROJECT = "huggingface"

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
DATA.mkdir(exist_ok=True)

api = wandb.Api()
entity = api.default_entity
matches: dict[str, list] = {}
for tag, rid in RUN_IDS.items():
    try:
        run = api.run(f"{entity}/{PROJECT}/{rid}")
        matches[tag] = [run]
        # W&B's _step counts wandb.log calls, NOT optimizer steps (the 7B run
        # shows _step 11364 with checkpoints ending at 403). The trainer step
        # is train/global_step.
        print(
            f"{tag}: {run.name} (id {run.id}, state {run.state}) "
            f"train/global_step={run.summary.get('train/global_step')} "
            f"train/epoch={run.summary.get('train/epoch')} _step={run.summary.get('_step')}"
        )
    except Exception as e:
        matches[tag] = []
        print(f"{tag}: run {rid} not fetchable: {e}")


def trainer_step_column(rows):
    for k in ("train/global_step", "global_step"):
        if any(k in r for r in rows):
            return k
    return None

# ---- R1: 7B training-reward series, raw rows --------------------------------
if not matches["7b"]:
    print("R1: the 7B run could not be fetched. Saying so plainly; no CSV written.")
else:
    run = matches["7b"][0]
    # scan_history returns EVERY logged row; run.history() samples ~500 points
    # and would violate the raw-rows requirement.
    rows = list(run.scan_history())
    step_col = trainer_step_column(rows)
    keep = {"_step", "train/epoch", "epoch"} | ({step_col} if step_col else set())
    keys: list[str] = []
    for r in rows:
        for k in r:
            if k not in keys and (k in keep or "reward" in k.lower()):
                keys.append(k)
    if "_step" not in keys:
        sys.exit("ABORT R1: no _step column in the run history — inspect the run by hand.")
    if step_col is None:
        print(
            "R1 WARNING: no train/global_step column in the history — the CSV "
            "carries only _step (a log-call counter, NOT the optimizer step); "
            "the figure's x-axis needs mapping before use."
        )
    keys.sort(key=lambda k: (k != "_step", k != step_col, k))  # steps first
    out_csv = DATA / "wandb_train_reward.csv"
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in keys})
    print(
        f"R1: wrote {out_csv} — {len(rows)} raw rows, columns {keys}, "
        f"run {run.url} (state {run.state}, final _step {run.summary.get('_step')})"
    )
    fig_script = ROOT / "figs" / "make_fig3_reward_curves.py"
    if fig_script.exists():
        print(f"R1: now run  python {fig_script.relative_to(ROOT)}  to produce figs/fig3_reward_curves.pdf")
    else:
        print(
            "R1: NOTE — figs/make_fig3_reward_curves.py does not exist in this "
            "working tree; the brief calls it 'already written'. The figure "
            "cannot be produced until that consumer lands."
        )

# ---- R2: 1.5B final step ----------------------------------------------------
if not matches["1.5b"]:
    print(
        "R2: the 1.5B run could not be fetched. Per the handoff: saying that "
        "plainly, and NOT inferring the number from the config. No JSON written."
    )
else:
    run = matches["1.5b"][0]
    final_step = run.summary.get("train/global_step")
    epochs = run.summary.get("train/epoch", run.summary.get("epoch"))
    source = "summary train/global_step"
    if final_step is None:
        # fall back to the last logged trainer step in the raw history
        hist = list(run.scan_history())
        col = trainer_step_column(hist)
        if col is None:
            sys.exit(
                "ABORT R2: neither the summary nor the history carries "
                "train/global_step — _step is a log-call counter and does NOT "
                "answer 403-vs-806; inspect the run by hand."
            )
        final_step = max(r[col] for r in hist if r.get(col) is not None)
        if epochs is None:
            epochs = max((r.get("train/epoch") for r in hist if r.get("train/epoch") is not None), default=None)
        source = f"max {col} over the raw scan_history rows"
    verdict = (
        f"{int(final_step)} — settled by {source} of W&B run {run.name} "
        f"(id {run.id}, state {run.state}, project {PROJECT}); chosen over two "
        f"same-named dead starts (failed, _step 1). W&B _step "
        f"({run.summary.get('_step')}) is a log-call counter and was not used."
    )
    out_json = DATA / "step_count_1p5b.json"
    out_json.write_text(
        json.dumps(
            {
                "final_step": int(final_step),
                "epochs_logged": epochs if epochs is not None else None,
                "wandb_run_url": run.url,
                "verdict": verdict,
            },
            indent=2,
        )
        + "\n"
    )
    print(f"R2: wrote {out_json} — final _step {final_step}, epochs_logged {epochs}, {run.url}")
