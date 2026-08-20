"""C1 distribution-shift measurements (aug21). Two subcommands.

  prepare  -- derive the canonical C1.1 train sample, assert it matches the
              prespecified c11_train_sample.json, and write the puzzle subset
              file for the eval harness. Run after `make data`.
  analyze  -- after the C1.1 eval has produced records: two-sample bootstrap
              of base-on-train-slice vs base-on-test (C1.1), and the train-vs-
              test stratum mix from the tagged DB (completes C1.3).

  python results-analysis/aug21/c1_shift_analysis.py prepare
  python results-analysis/aug21/c1_shift_analysis.py analyze
"""
import json, random, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
HERE = Path(__file__).parent
SPEC = json.loads((HERE / "c11_train_sample.json").read_text())

def prepare():
    from connections_rl.data.loader import load_puzzles, default_puzzles_path
    from connections_rl.data.splits import split_by_date
    puzzles = load_puzzles(default_puzzles_path())
    sp = split_by_date(puzzles)
    assert (len(sp.train), len(sp.val), len(sp.test)) == (807, 108, 162), sp.summary()
    train_ids = sorted(p.puzzle_id for p in sp.train)
    sample = sorted(random.Random(0).sample(train_ids, 162))
    assert sample == SPEC["sample_ids"], "derived sample differs from prespecified -- STOP"
    chosen = {p.puzzle_id: p for p in sp.train}
    recs = []
    for pid in sample:
        p = chosen[pid]
        recs.append({"id": p.puzzle_id, "puzzle_id": p.puzzle_id, "date": p.date,
                     "category": p.strata,
                     "answers": [{"level": g.level, "group": g.name, "members": list(g.members)}
                                 for g in p.groups]})
    out = HERE / "c11_train_slice_puzzles.json"
    out.write_text(json.dumps(recs))
    print(f"wrote {out} with {len(recs)} puzzles; sample matches prespecified list")

def _pctl(v, q):
    v = sorted(v); i = q * (len(v) - 1); lo = int(i); hi = min(lo + 1, len(v) - 1); f = i - lo
    return v[lo] * (1 - f) + v[hi] * f

def analyze():
    from collections import Counter
    from connections_rl.data.loader import load_puzzles, default_puzzles_path
    from connections_rl.data.splits import split_by_date
    def recs(p): return [json.loads(l) for l in open(p)]
    train_slice = recs(HERE / "c11-session/base-trainslice/records.jsonl")
    test = recs(HERE / "evalB-session/base/records.jsonl")
    a = [float(r["groups_correct"]) for r in train_slice]
    b = [float(r["groups_correct"]) for r in test]
    diff = sum(a)/len(a) - sum(b)/len(b)
    rng = random.Random(0); boots = []
    for _ in range(10000):
        ba = [a[rng.randrange(len(a))] for _ in range(len(a))]
        bb = [b[rng.randrange(len(b))] for _ in range(len(b))]
        boots.append(sum(ba)/len(ba) - sum(bb)/len(bb))
    result = {"c11": {"base_train_slice_mean_0to4": sum(a)/len(a), "n_train_slice": len(a),
                      "base_test_mean_0to4": sum(b)/len(b), "n_test": len(b),
                      "diff_train_minus_test": diff,
                      "two_sample_bootstrap95": [_pctl(boots, 0.025), _pctl(boots, 0.975)],
                      "B": 10000, "rng": "random.Random(0)",
                      "note": "two-sample (different puzzles; no pairing exists); both cells from the SAME aug21 vLLM session"}}
    sp = split_by_date(load_puzzles(default_puzzles_path()))
    tr = Counter(p.strata for p in sp.train); te = Counter(p.strata for p in sp.test)
    result["c13"] = {"train_counts": dict(tr), "train_props": {k: round(v/len(sp.train), 4) for k, v in tr.items()},
                     "test_counts": dict(te), "test_props": {k: round(v/len(sp.test), 4) for k, v in te.items()}}
    (HERE / "c1_shift_results.json").write_text(json.dumps(result, indent=1))
    print(json.dumps(result, indent=1))

if __name__ == "__main__":
    {"prepare": prepare, "analyze": analyze}[sys.argv[1]]()
