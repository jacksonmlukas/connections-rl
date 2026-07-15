"""``make eval`` / ``make eval-smoke``: run the offline evaluation.

Full mode reads configs/eval/default.yaml, evaluates each configured arm
against the held-out test split, saves per-arm results under results/<arm>/,
and prints paired comparisons.

Smoke mode (--smoke) needs no network, GPU, or external data: it evaluates
the bundled fixture puzzles with the deterministic oracle and random arms,
exercising the full harness + persistence + comparison path. CI runs it on
every push.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from connections_rl.data.loader import Puzzle, load_puzzles
from connections_rl.eval.harness import ArmResult, compare_arms, evaluate_arm, stratified_subsample
from connections_rl.eval.solvers import endpoint_solver, oracle_solver, random_solver

FIXTURES = Path(__file__).resolve().parents[3] / "tests" / "fixtures" / "puzzles_sample.json"


def _print_summary(result: ArmResult) -> None:
    s = result.summary()["OVERALL"]
    line = "  ".join(
        f"{m}={v[0]:.3f} [{v[1]:.3f}, {v[2]:.3f}]"
        for m, v in s.items()
        if m in ("solve_rate", "groups_correct", "invalid_rate")
    )
    print(f"{result.arm:>12}: n={len(result.records)}  {line}")


def run_smoke(out_dir: str = "results/smoke") -> None:
    puzzles = load_puzzles(FIXTURES)
    arms = {
        "oracle": evaluate_arm("oracle", oracle_solver(puzzles), puzzles),
        "random": evaluate_arm("random", random_solver(), puzzles),
    }
    for name, res in arms.items():
        res.save(Path(out_dir) / name)
        _print_summary(res)
    cmp = compare_arms(arms["oracle"], arms["random"])
    print(
        f"oracle vs random: McNemar p={cmp['mcnemar_p']:.4f}  diff CI={cmp['solve_rate_diff_ci']}"
    )
    assert arms["oracle"].summary()["OVERALL"]["solve_rate"][0] == 1.0, (
        "oracle must solve everything"
    )
    print("smoke eval OK")


def run_full(config_path: str) -> None:
    import yaml

    cfg = yaml.safe_load(Path(config_path).read_text())
    puzzles: list[Puzzle] = load_puzzles(cfg["puzzles"])
    if cfg.get("per_stratum"):
        puzzles = stratified_subsample(puzzles, cfg["per_stratum"], seed=cfg.get("seed", 0))
    results: dict[str, ArmResult] = {}
    for arm in cfg["arms"]:
        solver = endpoint_solver(
            model=arm["model"],
            base_url=arm.get("base_url"),
            temperature=arm.get("temperature", 0.0),
            max_tokens=arm.get("max_tokens", 1024),
        )
        res = evaluate_arm(arm["name"], solver, puzzles)
        res.save(Path(cfg.get("out_dir", "results")) / arm["name"], cfg.get("n_resamples", 1000))
        _print_summary(res)
        results[arm["name"]] = res
    names = list(results)
    comparisons = {}
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            comparisons[f"{a}_vs_{b}"] = compare_arms(results[a], results[b])
    out = Path(cfg.get("out_dir", "results")) / "comparisons.json"
    out.write_text(json.dumps(comparisons, indent=2, default=list))
    for k, v in comparisons.items():
        print(f"{k}: McNemar p={v['mcnemar_p']:.4f}  diff CI={v['solve_rate_diff_ci']}")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/eval/default.yaml")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args(argv)
    if args.smoke:
        run_smoke()
    else:
        run_full(args.config)


if __name__ == "__main__":
    main()
