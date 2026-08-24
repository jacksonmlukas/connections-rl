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
Edit ENTITY/PROJECT_CANDIDATES/NAME_HINTS if the naming differs.
"""

import csv
import json
import sys
from pathlib import Path

import wandb

NAME_HINTS = {"7b": "connections-rl-grpo-qwen7b-v1", "1.5b": "connections-rl-grpo-qwen1.5b-v2"}

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
DATA.mkdir(exist_ok=True)

api = wandb.Api()
entity = api.default_entity
# The training script sets run_name but never a W&B project, so the runs land
# in whatever project the logger defaulted to (TRL/transformers commonly use
# 'huggingface'). Enumerate ALL projects and search every one; print anything
# that even mentions connections/grpo so a near-miss name is visible.
projects = [p.name for p in api.projects(entity)]
print(f"W&B entity {entity!r}, projects: {projects}")
matches: dict[str, list] = {tag: [] for tag in NAME_HINTS}
near_misses = []
for proj in projects:
    try:
        runs = list(api.runs(f"{entity}/{proj}"))
    except Exception as e:
        print(f"project {proj!r}: {e}")
        continue
    for run in runs:
        nm = run.name or ""
        for tag, hint in NAME_HINTS.items():
            if hint in nm:
                matches[tag].append(run)
        if "connections" in nm.lower() or "grpo" in nm.lower():
            near_misses.append(
                f"  {proj}/{nm} (id {run.id}, state {run.state}, "
                f"final _step {run.summary.get('_step')})"
            )
print("runs mentioning connections/grpo across all projects:")
print("\n".join(near_misses) if near_misses else "  none")

for tag, ms in matches.items():
    if len(ms) > 1:
        sys.exit(
            f"ABORT: {len(ms)} runs match the {tag} hint "
            f"({[r.name + ' ' + r.id for r in ms]}); disambiguate NAME_HINTS "
            f"rather than letting the script pick one silently."
        )

# ---- R1: 7B training-reward series, raw rows --------------------------------
if not matches["7b"]:
    print(
        "R1: W&B does not have a run matching the 7B hint "
        f"{NAME_HINTS['7b']!r} in {PROJECT_CANDIDATES}. Saying so plainly; "
        "no CSV written."
    )
else:
    run = matches["7b"][0]
    # scan_history returns EVERY logged row; run.history() samples ~500 points
    # and would violate the raw-rows requirement.
    rows = list(run.scan_history())
    keys: list[str] = []
    for r in rows:
        for k in r:
            if k not in keys and (k == "_step" or "reward" in k.lower()):
                keys.append(k)
    if "_step" not in keys:
        sys.exit("ABORT R1: no _step column in the run history — inspect the run by hand.")
    keys.sort(key=lambda k: (k != "_step", k))  # _step first
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
        "R2: W&B does not have a run matching the 1.5B hint "
        f"{NAME_HINTS['1.5b']!r} in {PROJECT_CANDIDATES}. Per the handoff: "
        "saying that plainly, and NOT inferring the number from the config. "
        "No JSON written."
    )
else:
    run = matches["1.5b"][0]
    final_step = run.summary.get("_step")
    epochs = run.summary.get("train/epoch", run.summary.get("epoch"))
    if final_step is None:
        sys.exit("ABORT R2: run found but summary has no _step — inspect the run by hand.")
    verdict = (
        f"{int(final_step)} — settled by the final logged _step of the W&B run "
        f"({run.name}, state {run.state}); the config's epochs field does not "
        f"override what was actually logged."
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
