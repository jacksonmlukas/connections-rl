"""Leakage-aware train/val/test splits.

Puzzles are split strictly by publication date: everything the model trains on
predates everything it is validated and tested on. NYT reuses themes and
words across puzzles, so a random split would leak.
"""

from __future__ import annotations

from dataclasses import dataclass

from connections_rl.data.loader import Puzzle


@dataclass(frozen=True)
class DateSplit:
    train: list[Puzzle]
    val: list[Puzzle]
    test: list[Puzzle]

    def summary(self) -> dict[str, dict[str, str | int]]:
        def block(ps: list[Puzzle]) -> dict[str, str | int]:
            return {
                "n": len(ps),
                "start": ps[0].date if ps else "",
                "end": ps[-1].date if ps else "",
            }

        return {"train": block(self.train), "val": block(self.val), "test": block(self.test)}


def split_by_date(
    puzzles: list[Puzzle], val_frac: float = 0.10, test_frac: float = 0.15
) -> DateSplit:
    """Chronological split. The most recent ``test_frac`` of puzzles is test,
    the ``val_frac`` before that is validation, the rest is train.

    Ties on date never straddle a boundary: boundaries are moved earlier so a
    calendar date appears in exactly one split.
    """
    if not 0 < val_frac + test_frac < 1:
        raise ValueError("val_frac + test_frac must be in (0, 1)")
    ordered = sorted(puzzles, key=lambda p: (p.date, p.puzzle_id))
    n = len(ordered)
    test_start = n - max(1, int(round(n * test_frac)))
    val_start = test_start - max(1, int(round(n * val_frac)))

    def push_back(idx: int) -> int:
        # Move a boundary earlier until it lands on a date change.
        while 0 < idx < n and ordered[idx].date == ordered[idx - 1].date:
            idx -= 1
        return idx

    test_start = push_back(test_start)
    val_start = push_back(min(val_start, test_start))
    if val_start <= 0 or test_start <= val_start:
        raise ValueError("not enough puzzles for the requested split fractions")
    return DateSplit(
        train=ordered[:val_start], val=ordered[val_start:test_start], test=ordered[test_start:]
    )
