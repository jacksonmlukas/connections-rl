"""Offline evaluation harness.

An *arm* is anything that maps a puzzle prompt to a raw completion string
(base model, SFT, GRPO, or a replayed multi-agent baseline). The harness
scores each completion with the same verifiable reward used in training,
aggregates per stratum with bootstrap CIs, and runs paired significance
tests between arms on the identical puzzle set.
"""

from __future__ import annotations

import dataclasses
import json
import random
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from connections_rl.data.formatting import build_chat
from connections_rl.data.loader import Puzzle
from connections_rl.eval.stats import bootstrap_ci, mcnemar_exact, paired_bootstrap_diff
from connections_rl.reward.reward import RewardConfig, reward_breakdown

# A solver receives the chat messages and returns the raw model completion.
Solver = Callable[[list[dict[str, str]]], str]


@dataclass(frozen=True)
class PuzzleRecord:
    puzzle_id: int
    date: str
    strata: str
    solved: bool
    groups_correct: int
    one_away: bool  # 3/4 groups correct and a fourth group one word off
    invalid_format: bool
    reward: float
    latency_ms: float = 0.0


@dataclass
class ArmResult:
    arm: str
    records: list[PuzzleRecord] = field(default_factory=list)

    # ---- metrics -------------------------------------------------------
    def metric_values(self, metric: str) -> list[float]:
        pick: dict[str, Callable[[PuzzleRecord], float]] = {
            "solve_rate": lambda r: float(r.solved),
            "groups_correct": lambda r: float(r.groups_correct),
            "one_away_rate": lambda r: float(r.one_away),
            "invalid_rate": lambda r: float(r.invalid_format),
            "reward": lambda r: r.reward,
        }
        return [pick[metric](r) for r in self.records]

    def summary(
        self, n_resamples: int = 1000, seed: int = 0
    ) -> dict[str, dict[str, tuple[float, float, float]]]:
        """{stratum: {metric: (mean, lo, hi)}}; stratum 'OVERALL' first."""
        strata: dict[str, list[PuzzleRecord]] = {"OVERALL": list(self.records)}
        for r in self.records:
            strata.setdefault(r.strata, []).append(r)
        out: dict[str, dict[str, tuple[float, float, float]]] = {}
        for name, records in strata.items():
            sub = ArmResult(self.arm, records)
            out[name] = {
                m: bootstrap_ci(sub.metric_values(m), n_resamples=n_resamples, seed=seed)
                for m in ("solve_rate", "groups_correct", "one_away_rate", "invalid_rate", "reward")
            }
        return out

    def groups_correct_distribution(self) -> dict[int, int]:
        dist = {k: 0 for k in range(5)}
        for r in self.records:
            dist[r.groups_correct] += 1
        return dist

    # ---- persistence ---------------------------------------------------
    def save(self, out_dir: str | Path, n_resamples: int = 1000) -> None:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        with open(out / "records.jsonl", "w", encoding="utf-8") as f:
            for r in self.records:
                f.write(json.dumps(dataclasses.asdict(r)) + "\n")
        with open(out / "metrics.json", "w", encoding="utf-8") as f:
            json.dump(
                {
                    "arm": self.arm,
                    "n": len(self.records),
                    "n_resamples": n_resamples,
                    "summary": self.summary(n_resamples=n_resamples),
                    "groups_correct_distribution": self.groups_correct_distribution(),
                },
                f,
                indent=2,
            )

    @classmethod
    def load_records(cls, path: str | Path, arm: str = "") -> ArmResult:
        records = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    records.append(PuzzleRecord(**json.loads(line)))
        return cls(arm=arm or Path(path).parent.name, records=records)


def stratified_subsample(
    puzzles: Sequence[Puzzle], per_stratum: int, seed: int = 0
) -> list[Puzzle]:
    """Draw up to ``per_stratum`` puzzles from each stratum, deterministically."""
    rng = random.Random(seed)
    by_stratum: dict[str, list[Puzzle]] = {}
    for p in puzzles:
        by_stratum.setdefault(p.strata, []).append(p)
    sample: list[Puzzle] = []
    for _, pool in sorted(by_stratum.items()):
        pool = sorted(pool, key=lambda p: p.puzzle_id)
        rng.shuffle(pool)
        sample.extend(pool[: min(per_stratum, len(pool))])
    return sorted(sample, key=lambda p: p.puzzle_id)


def evaluate_arm(
    arm: str,
    solver: Solver,
    puzzles: Sequence[Puzzle],
    reward_config: RewardConfig | None = None,
    prompt_seed: int | None = None,
) -> ArmResult:
    """Run ``solver`` over ``puzzles`` and score every completion.

    Solver exceptions are recorded as invalid completions rather than
    aborting the run (matches gvc-local's harness behaviour).
    """
    if reward_config is None:
        reward_config = RewardConfig()
    result = ArmResult(arm=arm)
    for p in puzzles:
        start = time.perf_counter()
        try:
            completion = solver(build_chat(p, seed=prompt_seed))
        except Exception:
            completion = ""
        latency_ms = (time.perf_counter() - start) * 1000
        br = reward_breakdown(completion, p.words, p.answer_sets, reward_config)
        result.records.append(
            PuzzleRecord(
                puzzle_id=p.puzzle_id,
                date=p.date,
                strata=p.strata,
                solved=br.solved,
                groups_correct=br.correct_groups,
                one_away=br.correct_groups == 3 and br.one_away_groups >= 1,
                invalid_format=not br.valid,
                reward=br.total,
                latency_ms=latency_ms,
            )
        )
    return result


def compare_arms(
    a: ArmResult, b: ArmResult, n_resamples: int = 1000, seed: int = 0
) -> dict[str, float | tuple[float, float, float] | int]:
    """Paired comparison of two arms evaluated on the SAME puzzles.

    Returns McNemar exact p on solved/unsolved plus a paired-bootstrap CI on
    the solve-rate difference (a - b).
    """
    ra = {r.puzzle_id: r for r in a.records}
    rb = {r.puzzle_id: r for r in b.records}
    common = sorted(set(ra) & set(rb))
    if not common:
        raise ValueError("arms share no puzzles; comparison must be paired")
    a_solved = [float(ra[i].solved) for i in common]
    b_solved = [float(rb[i].solved) for i in common]
    only_a = sum(1 for x, y in zip(a_solved, b_solved, strict=True) if x > y)
    only_b = sum(1 for x, y in zip(a_solved, b_solved, strict=True) if y > x)
    diff = paired_bootstrap_diff(a_solved, b_solved, n_resamples=n_resamples, seed=seed)
    return {
        "n_paired": len(common),
        "a_solve_rate": sum(a_solved) / len(common),
        "b_solve_rate": sum(b_solved) / len(common),
        "mcnemar_p": mcnemar_exact(only_a, only_b),
        "solve_rate_diff_ci": diff,
        "discordant_a_only": only_a,
        "discordant_b_only": only_b,
    }
