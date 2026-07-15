"""Deterministic, verifiable reward for NYT Connections completions.

Design (per project plan):
  * format reward       — small credit for a structurally valid answer
                          (4 groups x 4 words, all 16 board words exactly once);
  * grouping reward     — fully-correct groups / 4, scaled;
  * solve bonus         — extra credit when all 4 groups are correct;
  * one-away shaping    — optional small credit per group with exactly 3
                          correct words (smooths early GRPO learning);
  * invalid penalty     — invalid structure earns a flat (default negative)
                          score, which discourages reward hacking via malformed
                          output.

The reward never sees test-set puzzles: training code only receives train/val
splits from ``connections_rl.data``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from connections_rl.data.loader import normalize_word

_ANSWER_BLOCK = re.compile(r"<ANSWER>(.*?)</ANSWER>", re.DOTALL | re.IGNORECASE)
_GROUP_LINE = re.compile(r"^\s*Group\s*\d*\s*[:\-]\s*(.+?)\s*$", re.IGNORECASE | re.MULTILINE)


@dataclass(frozen=True)
class RewardConfig:
    format_reward: float = 0.1
    grouping_weight: float = 1.0
    solve_bonus: float = 0.5
    one_away_credit: float = 0.05  # per group with exactly 3 correct words
    invalid_penalty: float = -0.1  # flat score for structurally invalid output

    @property
    def max_reward(self) -> float:
        return self.format_reward + self.grouping_weight + self.solve_bonus


@dataclass(frozen=True)
class RewardBreakdown:
    valid: bool
    correct_groups: int  # 0..4
    one_away_groups: int  # groups with exactly 3/4 correct
    solved: bool
    total: float


def parse_groups(text: str) -> list[frozenset[str]] | None:
    """Extract 4 groups of 4 words from a completion, or None.

    Prefers the LAST <ANSWER>...</ANSWER> block (the model may reason first);
    falls back to the last four ``Group N: ...`` lines anywhere in the text.
    Words are normalized to uppercase. Structure is NOT validated against a
    board here — see :func:`validate_against_board`.
    """
    blocks = _ANSWER_BLOCK.findall(text)
    scope = blocks[-1] if blocks else text
    lines = _GROUP_LINE.findall(scope)
    if len(lines) < 4:
        return None
    groups = []
    for line in lines[-4:]:
        words = [normalize_word(w) for w in line.split(",") if w.strip()]
        if len(words) != 4 or len(set(words)) != 4:
            return None
        groups.append(frozenset(words))
    return groups


def validate_against_board(groups: list[frozenset[str]], board: list[str]) -> bool:
    """True iff the 4 groups tile the 16 board words exactly."""
    used: set[str] = set()
    for g in groups:
        if used & g:
            return False
        used |= g
    return used == {normalize_word(w) for w in board}


def score_groups(
    predicted: list[frozenset[str]], answer_sets: list[frozenset[str]]
) -> tuple[int, int]:
    """Return (fully correct groups, one-away groups).

    A predicted group is one-away if it shares exactly 3 words with some
    answer group. Each answer group is credited at most once.
    """
    remaining = list(answer_sets)
    correct = 0
    for g in predicted:
        if g in remaining:
            correct += 1
            remaining.remove(g)
    one_away = 0
    unmatched_answers = list(remaining)
    for g in predicted:
        if g in answer_sets:
            continue
        for ans in unmatched_answers:
            if len(g & ans) == 3:
                one_away += 1
                unmatched_answers.remove(ans)
                break
    return correct, one_away


def reward_breakdown(
    completion: str,
    board: list[str],
    answer_sets: list[frozenset[str]],
    config: RewardConfig | None = None,
) -> RewardBreakdown:
    if config is None:
        config = RewardConfig()
    groups = parse_groups(completion)
    if groups is None or not validate_against_board(groups, board):
        return RewardBreakdown(
            valid=False,
            correct_groups=0,
            one_away_groups=0,
            solved=False,
            total=config.invalid_penalty,
        )
    correct, one_away = score_groups(groups, answer_sets)
    solved = correct == 4
    total = (
        config.format_reward
        + config.grouping_weight * (correct / 4)
        + (config.solve_bonus if solved else 0.0)
        + config.one_away_credit * one_away
    )
    return RewardBreakdown(
        valid=True, correct_groups=correct, one_away_groups=one_away, solved=solved, total=total
    )


def reward(
    completion: str,
    board: list[str],
    answer_sets: list[frozenset[str]],
    config: RewardConfig | None = None,
) -> float:
    return reward_breakdown(completion, board, answer_sets, config).total
