"""TASK D paired bootstrap: step-50 vs base on groups-correct (0-4), within-session.

Run AFTER the eval in TASKD_RUNBOOK.md has written
results-analysis/aug20/taskD-session/{base,grpo-ckpt50,grpo-final}/records.jsonl.

Uses the repo's own estimator (eval/stats.py::paired_bootstrap_diff) at its
committed default bootstrap seed 0, matching the existing methodology.

    python results-analysis/aug20/taskD_paired_groups.py
"""
import json, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from connections_rl.eval.stats import paired_bootstrap_diff  # noqa: E402

SESSION = Path(__file__).parent / "taskD-session"

def load(arm):
    return {r["puzzle_id"]: r for r in map(json.loads, open(SESSION / arm / "records.jsonl"))}

arms = {a: load(a) for a in ("base", "grpo-ckpt50", "grpo-final")}
ids = sorted(set.intersection(*(set(v) for v in arms.values())))
out = {"n_paired": len(ids), "bootstrap_seed": 0, "scale": "0-4 groups-correct",
       "session": "taskD-session (all arms served in one vLLM session)", "comparisons": {}}
for a, b in (("grpo-ckpt50", "base"), ("grpo-final", "base"), ("grpo-ckpt50", "grpo-final")):
    va = [float(arms[a][i]["groups_correct"]) for i in ids]
    vb = [float(arms[b][i]["groups_correct"]) for i in ids]
    out["comparisons"][f"{a}_minus_{b}"] = paired_bootstrap_diff(va, vb)
Path(SESSION / "paired_groups.json").write_text(json.dumps(out, indent=1, default=list))
print(json.dumps(out, indent=1, default=list))
