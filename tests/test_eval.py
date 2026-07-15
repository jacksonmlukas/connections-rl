"""Eval harness + stats tests: determinism, CIs, significance, persistence."""

import math
from pathlib import Path

import pytest

from connections_rl.data.loader import Puzzle
from connections_rl.eval.harness import compare_arms, evaluate_arm, stratified_subsample
from connections_rl.eval.solvers import oracle_solver, random_solver
from connections_rl.eval.stats import bootstrap_ci, ece, mcnemar_exact, paired_bootstrap_diff

# ---------- stats primitives ----------


def test_bootstrap_ci_contains_mean() -> None:
    vals = [0.0, 1.0] * 50
    mean, lo, hi = bootstrap_ci(vals, n_resamples=500, seed=0)
    assert mean == pytest.approx(0.5)
    assert lo <= mean <= hi
    assert 0.3 < lo and hi < 0.7


def test_bootstrap_ci_deterministic() -> None:
    vals = [float(i % 3) for i in range(30)]
    assert bootstrap_ci(vals, seed=7) == bootstrap_ci(vals, seed=7)
    assert bootstrap_ci(vals, seed=7) != bootstrap_ci(vals, seed=8)


def test_bootstrap_ci_degenerate() -> None:
    mean, lo, hi = bootstrap_ci([1.0] * 10)
    assert (mean, lo, hi) == (1.0, 1.0, 1.0)
    with pytest.raises(ValueError):
        bootstrap_ci([])


def test_paired_bootstrap_detects_difference() -> None:
    a = [1.0] * 40 + [0.0] * 10
    b = [0.0] * 40 + [1.0] * 10
    diff, lo, hi = paired_bootstrap_diff(a, b, n_resamples=500)
    assert diff == pytest.approx(0.6)
    assert lo > 0  # CI excludes zero


def test_mcnemar_exact_values() -> None:
    assert mcnemar_exact(0, 0) == 1.0
    assert mcnemar_exact(5, 5) == 1.0
    # b=10, c=0 -> two-sided exact p = 2 * (1/2)^10
    assert mcnemar_exact(10, 0) == pytest.approx(2 * 0.5**10)
    assert mcnemar_exact(10, 0) < 0.05
    assert 0 < mcnemar_exact(7, 2) <= 1.0


def test_ece_perfect_calibration() -> None:
    conf = [0.25] * 4 + [0.75] * 4
    out = [1, 0, 0, 0, 1, 1, 1, 0]
    err, bins = ece(conf, out, n_bins=4)
    assert err == pytest.approx(0.0)
    assert sum(b["n"] for b in bins) == 8


def test_ece_miscalibrated() -> None:
    err, _ = ece([0.9] * 10, [0] * 10, n_bins=10)
    assert err == pytest.approx(0.9)
    assert not math.isnan(err)


# ---------- harness ----------


def test_oracle_solves_everything(puzzles: list[Puzzle]) -> None:
    res = evaluate_arm("oracle", oracle_solver(puzzles), puzzles)
    s = res.summary()["OVERALL"]
    assert s["solve_rate"][0] == 1.0
    assert s["invalid_rate"][0] == 0.0
    assert res.groups_correct_distribution()[4] == len(puzzles)


def test_random_solver_valid_but_weak(puzzles: list[Puzzle]) -> None:
    res = evaluate_arm("random", random_solver(), puzzles)
    s = res.summary()["OVERALL"]
    assert s["invalid_rate"][0] == 0.0  # random partitions are structurally valid
    assert s["solve_rate"][0] < 1.0


def test_crashing_solver_recorded_as_invalid(puzzles: list[Puzzle]) -> None:
    def boom(messages: list[dict[str, str]]) -> str:
        raise RuntimeError("endpoint down")

    res = evaluate_arm("boom", boom, puzzles)
    assert res.summary()["OVERALL"]["invalid_rate"][0] == 1.0


def test_summary_has_strata(puzzles: list[Puzzle]) -> None:
    res = evaluate_arm("oracle", oracle_solver(puzzles), puzzles)
    summary = res.summary()
    assert "OVERALL" in summary
    assert {p.strata for p in puzzles} <= set(summary)


def test_stratified_subsample(puzzles: list[Puzzle]) -> None:
    sample = stratified_subsample(puzzles, per_stratum=1, seed=0)
    strata = [p.strata for p in sample]
    assert len(strata) == len(set(strata))  # one per stratum
    assert stratified_subsample(puzzles, 1, seed=0) == stratified_subsample(puzzles, 1, seed=0)


def test_compare_arms_oracle_beats_random(puzzles: list[Puzzle]) -> None:
    oracle = evaluate_arm("oracle", oracle_solver(puzzles), puzzles)
    rand = evaluate_arm("random", random_solver(), puzzles)
    cmp = compare_arms(oracle, rand)
    assert cmp["n_paired"] == len(puzzles)
    assert cmp["a_solve_rate"] == 1.0
    diff_ci = cmp["solve_rate_diff_ci"]
    assert diff_ci[0] > 0


def test_results_persistence_round_trip(tmp_path: Path, puzzles: list[Puzzle]) -> None:
    from connections_rl.eval.harness import ArmResult

    res = evaluate_arm("oracle", oracle_solver(puzzles), puzzles)
    res.save(tmp_path / "oracle")
    loaded = ArmResult.load_records(tmp_path / "oracle" / "records.jsonl")
    assert loaded.arm == "oracle"
    assert [r.puzzle_id for r in loaded.records] == [r.puzzle_id for r in res.records]
    assert (tmp_path / "oracle" / "metrics.json").exists()


def test_smoke_eval_runs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from connections_rl.eval.run import run_smoke

    run_smoke(out_dir=str(tmp_path / "smoke"))
    assert (tmp_path / "smoke" / "oracle" / "metrics.json").exists()


def test_gate(tmp_path: Path, puzzles: list[Puzzle]) -> None:
    from connections_rl.eval.gate import main as gate_main

    good = evaluate_arm("grpo", oracle_solver(puzzles), puzzles)
    bad = evaluate_arm("sft", random_solver(), puzzles)
    good.save(tmp_path / "grpo")
    bad.save(tmp_path / "sft")
    # candidate (oracle) vs baseline (random): pass
    assert (
        gate_main(
            [
                "--candidate",
                str(tmp_path / "grpo/metrics.json"),
                "--baseline",
                str(tmp_path / "sft/metrics.json"),
            ]
        )
        == 0
    )
    # candidate (random) vs baseline (oracle): fail — random < oracle's CI lower bound (1.0)
    assert (
        gate_main(
            [
                "--candidate",
                str(tmp_path / "sft/metrics.json"),
                "--baseline",
                str(tmp_path / "grpo/metrics.json"),
            ]
        )
        == 1
    )
    # missing file: skip
    assert (
        gate_main(
            [
                "--candidate",
                str(tmp_path / "nope.json"),
                "--baseline",
                str(tmp_path / "grpo/metrics.json"),
            ]
        )
        == 2
    )
