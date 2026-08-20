"""aug21 paired bootstraps on groups-correct (0-4), repo estimator, seed 0.

Covers B2 (step-100 vs base / vs step-50) and B1 (noscale-final vs base,
noscale-final vs grpo-final, noscale-peak vs base), all within the aug21
eval session. Run after taskB2_eval_test.yaml has produced records.

  python results-analysis/aug21/taskB2_paired.py
"""
import json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from connections_rl.eval.stats import paired_bootstrap_diff

S = Path(__file__).parent / "evalB-session"
ARMS = ["base", "grpo-ckpt50", "grpo-ckpt100", "grpo-final", "noscale-final", "noscale-peak"]
def load(a): return {json.loads(l)["puzzle_id"]: json.loads(l) for l in open(S / a / "records.jsonl")}
have = {a: load(a) for a in ARMS if (S / a / "records.jsonl").exists()}
ids = sorted(set.intersection(*(set(v) for v in have.values())))
out = {"n_paired": len(ids), "bootstrap_seed": 0, "scale": "0-4 groups-correct",
       "session": "aug21 evalB-session (single vLLM session)", "arms_present": sorted(have), "comparisons": {}}
PAIRS = [("grpo-ckpt100", "base"), ("grpo-ckpt100", "grpo-ckpt50"), ("grpo-ckpt50", "base"),
         ("noscale-final", "base"), ("noscale-final", "grpo-final"), ("noscale-peak", "base")]
for a, b in PAIRS:
    if a in have and b in have:
        va = [float(have[a][i]["groups_correct"]) for i in ids]
        vb = [float(have[b][i]["groups_correct"]) for i in ids]
        out["comparisons"][f"{a}_minus_{b}"] = paired_bootstrap_diff(va, vb)
(S / "paired_groups.json").write_text(json.dumps(out, indent=1, default=list))
print(json.dumps(out, indent=1, default=list))
