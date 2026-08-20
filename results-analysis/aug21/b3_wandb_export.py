"""B3 + the 1.5B step-count question (aug21). Run LOCALLY with your W&B login.

Exports the training-reward series for the reported 7B run and the 1.5B v2 run
as JSON, and prints each run's final step -- settling 403-vs-806 for 1.5B.

  pip install wandb && python results-analysis/aug21/b3_wandb_export.py
Edit ENTITY/RUN_FILTERS if the project naming differs.
"""
import json
from pathlib import Path
import wandb

ENTITY = None  # default entity
PROJECT_CANDIDATES = ["connections-rl"]
NAME_HINTS = {"7b": "connections-rl-grpo-qwen7b-v1", "1.5b": "connections-rl-grpo-qwen1.5b-v2"}

api = wandb.Api()
out = {}
for proj in PROJECT_CANDIDATES:
    try:
        runs = api.runs(f"{ENTITY + '/' if ENTITY else ''}{proj}")
    except Exception as e:
        print(proj, "->", e); continue
    for run in runs:
        for tag, hint in NAME_HINTS.items():
            if hint in (run.name or ""):
                hist = run.history(keys=["train/reward", "train/rewards/mean", "reward", "_step"], pandas=False)
                series = [{k: row.get(k) for k in row} for row in hist]
                out[tag] = {"run_name": run.name, "state": run.state,
                            "last_step": run.summary.get("_step"),
                            "n_history_rows": len(series), "history": series}
                print(tag, run.name, "final _step =", run.summary.get("_step"))
Path(__file__).parent.joinpath("b3_training_reward.json").write_text(json.dumps(out, indent=1))
print("wrote b3_training_reward.json -- the 1.5b final step answers the open question")
