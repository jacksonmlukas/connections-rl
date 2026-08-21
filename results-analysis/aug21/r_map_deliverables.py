"""R-handoff (aug21): map session outputs to the EXACT data/ paths named in
the brief, with verification gates. Run after the aug21 eval session's outputs
exist (in-session on the GPU box, or locally once the Hub mirror is pulled).

  python results-analysis/aug21/r_map_deliverables.py

Writes (only what its inputs support; missing inputs are reported, never
substituted):
  data/b1_noscale_test_metrics.json   (R3b: noscale-final on 162 test)
  data/b1_noscale_ckpt_curve.json     (R3b: entropy/KL/structural/semantic series)
  data/b1_noscale_peak_test.json      (R3b: val-peak checkpoint on test, + step)
  data/b1_resolved_config.yaml        (R3b: resolved eval config, substituted ckpt)
  data/b2_step100_test.json           (R4: original run step-100 on test + paired)
  data/test_generations_{base,ckpt50,final}.jsonl   (R6: gated on exact
        reproduction of 26 / 77 / 4 recovered groups of 648 — a mismatch is
        reported and the file is NOT shipped)

Every number is read from the session artifacts (metrics.json / records.jsonl
/ paired_groups.json); nothing is re-estimated here except the recovered-group
counts, which are exact integer sums over records.
"""

import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

AUG21 = ROOT / "results-analysis" / "aug21"
S = AUG21 / "evalB-session"
DATA = ROOT / "data"
DATA.mkdir(exist_ok=True)

SESSION_NOTE = "aug21 evalB-session (single vLLM session; not comparable across sessions)"
# Expected recovered-group counts (of 648) for the R6 gate — session A and the
# aug20 Task D session agree on these exactly (0/162 per-puzzle diffs):
R6_EXPECTED = {"base": 26, "ckpt50": 77, "final": 4}
R6_ARM_DIRS = {"base": "base", "ckpt50": "grpo-ckpt50", "final": "grpo-final"}

problems: list[str] = []


def arm_block(arm_dir: str) -> dict | None:
    """Test-split metrics for one arm, read off the session artifacts."""
    mpath = S / arm_dir / "metrics.json"
    rpath = S / arm_dir / "records.jsonl"
    if not (mpath.exists() and rpath.exists()):
        problems.append(f"missing {mpath} / {rpath} — arm {arm_dir!r} not mapped")
        return None
    metrics = json.loads(mpath.read_text())
    records = [json.loads(line) for line in rpath.read_text().splitlines() if line.strip()]
    n = len(records)
    overall = metrics["summary"]["OVERALL"]
    recovered = sum(int(r["groups_correct"]) for r in records)
    return {
        "arm": arm_dir,
        "n_puzzles": n,
        "scale": "0-4 groups-correct per puzzle",
        "session": SESSION_NOTE,
        "groups_correct_mean": overall["groups_correct"][0],
        "groups_correct_ci95": overall["groups_correct"][1:3],
        "recovered_groups_count": recovered,
        "recovered_groups_denominator": 4 * n,
        "invalid_rate": overall["invalid_rate"][0],
        "invalid_count": sum(int(bool(r["invalid_format"])) for r in records),
        "reward_mean": overall["reward"][0],
        "solve_count": sum(int(bool(r["solved"])) for r in records),
        "bootstrap": {"n_resamples": metrics.get("n_resamples"), "seed": 0},
        "groups_correct_distribution": metrics.get("groups_correct_distribution"),
        "source": [str(mpath.relative_to(ROOT)), str(rpath.relative_to(ROOT))],
    }


def paired() -> dict:
    p = S / "paired_groups.json"
    if not p.exists():
        problems.append(f"missing {p} — run taskB2_paired.py first; paired fields omitted")
        return {}
    return json.loads(p.read_text()).get("comparisons", {})


# ---- R3b: noscale-final test metrics ---------------------------------------
blk = arm_block("noscale-final")
if blk is not None:
    (DATA / "b1_noscale_test_metrics.json").write_text(json.dumps(blk, indent=2) + "\n")
    print(f"wrote data/b1_noscale_test_metrics.json — {blk['recovered_groups_count']}"
          f"/{blk['recovered_groups_denominator']} groups recovered")

# ---- R3b: checkpoint curve (same shape as entropy-kl-7b.json) ---------------
src_curve = AUG21 / "entropy-kl-7b-noscale.json"
if src_curve.exists():
    shutil.copyfile(src_curve, DATA / "b1_noscale_ckpt_curve.json")
    print("wrote data/b1_noscale_ckpt_curve.json")
else:
    problems.append(f"missing {src_curve} — b1_noscale_ckpt_curve.json not mapped")

# ---- R3b: val-peak checkpoint on test + which step --------------------------
peak_step = None
val_curve_path = AUG21 / "ckpt-curve-7b-noscale.json"
if val_curve_path.exists():
    pts = json.loads(val_curve_path.read_text())
    peak = max(pts, key=lambda p: p["semantic_groups_correct"])
    peak_step = peak["step"]
blk = arm_block("noscale-peak")
if blk is not None:
    blk["peak_step"] = peak_step
    blk["peak_selection"] = (
        "argmax of validation semantic score on ckpt-curve-7b-noscale.json, "
        "frozen before any test puzzle was scored"
        if peak_step is not None
        else "PEAK STEP SOURCE MISSING — ckpt-curve-7b-noscale.json not found"
    )
    if peak_step is None:
        problems.append("ckpt-curve-7b-noscale.json missing — peak_step is null in "
                        "b1_noscale_peak_test.json")
    (DATA / "b1_noscale_peak_test.json").write_text(json.dumps(blk, indent=2) + "\n")
    print(f"wrote data/b1_noscale_peak_test.json (peak step {peak_step})")

# ---- R3b: resolved config ----------------------------------------------------
resolved = AUG21 / "taskB2_eval_test.resolved.yaml"
if resolved.exists():
    shutil.copyfile(resolved, DATA / "b1_resolved_config.yaml")
    print("wrote data/b1_resolved_config.yaml")
else:
    problems.append(f"missing {resolved} — b1_resolved_config.yaml not mapped")

# ---- R4: step-100 of the ORIGINAL run on test --------------------------------
blk = arm_block("grpo-ckpt100")
if blk is not None:
    cmp = paired()
    blk["paired_vs_base"] = cmp.get("grpo-ckpt100_minus_base")
    blk["paired_vs_step50"] = cmp.get("grpo-ckpt100_minus_grpo-ckpt50")
    blk["paired_note"] = ("paired bootstrap on 0-4 groups-correct, seed 0, "
                          "within the aug21 session; from evalB-session/paired_groups.json")
    (DATA / "b2_step100_test.json").write_text(json.dumps(blk, indent=2) + "\n")
    print(f"wrote data/b2_step100_test.json — {blk['recovered_groups_count']}"
          f"/{blk['recovered_groups_denominator']} groups recovered")

# ---- R6: generation captures, gated -----------------------------------------
from connections_rl.data.loader import load_puzzles  # noqa: E402
from connections_rl.reward.reward import reward_breakdown  # noqa: E402

test_path = ROOT / "data" / "splits" / "puzzles_test.json"
if not test_path.exists():
    problems.append(f"missing {test_path} — run `make data`; R6 gate cannot run")
else:
    pz = {p.puzzle_id: p for p in load_puzzles(test_path)}
    for tag, arm_dir in R6_ARM_DIRS.items():
        gpath = S / arm_dir / "generations.jsonl"
        if not gpath.exists():
            problems.append(f"missing {gpath} — test_generations_{tag}.jsonl not shipped")
            continue
        rows = [json.loads(line) for line in gpath.read_text().splitlines() if line.strip()]
        row_mismatch = 0
        total = 0
        for row in rows:
            p = pz[row["puzzle_id"]]
            br = reward_breakdown(row["generation"], p.words, p.answer_sets)
            if br.correct_groups != row["groups_correct"] or br.valid != row["valid"]:
                row_mismatch += 1
            total += br.correct_groups
        expected = R6_EXPECTED[tag]
        if row_mismatch or len(rows) != 162 or total != expected:
            problems.append(
                f"R6 GATE FAILED for {tag}: n={len(rows)} (need 162), "
                f"{row_mismatch} rows where recomputation disagrees with the "
                f"stored score, recomputed total {total}/648 vs expected "
                f"{expected}/648. NOT shipping test_generations_{tag}.jsonl — "
                f"per the brief, the capture is not of the same generations "
                f"and the row is worthless; saying so instead."
            )
            continue
        shutil.copyfile(gpath, DATA / f"test_generations_{tag}.jsonl")
        print(f"wrote data/test_generations_{tag}.jsonl — recomputed {total}/648 "
              f"== expected {expected}/648, all {len(rows)} rows self-consistent")

# ---- summary -----------------------------------------------------------------
print()
if problems:
    print("NOT DELIVERED (reported plainly, nothing substituted):")
    for p in problems:
        print(" -", p)
    sys.exit(1)
print("all mapped deliverables written to data/")
