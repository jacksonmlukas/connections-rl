"""Load the NYT Connections puzzle DB (gvc-local schema).

The canonical source is gvc-local's ``data/puzzles/tagged_connections.json``:
a JSON array of records like::

    {
      "id": 1,
      "date": "2023-06-12",
      "answers": [
        {"level": 0, "group": "WET WEATHER", "members": ["HAIL","RAIN","SLEET","SNOW"]},
        ...
      ],
      "puzzle_id": 1,
      "category": "wordplay"   # strata tag (tagged file only)
    }

We parse that schema directly (it is tiny) so the core package has no runtime
dependency on gvc-local. The gvc-local repo remains the source of truth for
the data and the multi-agent baseline numbers.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_STRATUM = "uncategorised"
PUZZLES_ENV_VAR = "CONNECTIONS_PUZZLES"


@dataclass(frozen=True)
class Group:
    """One answer group: 4 words plus NYT difficulty level (0=yellow .. 3=purple)."""

    level: int
    name: str
    members: tuple[str, ...]


@dataclass(frozen=True)
class Puzzle:
    puzzle_id: int
    date: str  # ISO YYYY-MM-DD; sortable lexicographically
    groups: tuple[Group, ...]
    strata: str = DEFAULT_STRATUM

    @property
    def words(self) -> list[str]:
        return [w for g in self.groups for w in g.members]

    @property
    def answer_sets(self) -> list[frozenset[str]]:
        return [frozenset(normalize_word(w) for w in g.members) for g in self.groups]


@dataclass
class PuzzleValidationError(Exception):
    record: dict = field(default_factory=dict)
    reason: str = ""

    def __str__(self) -> str:
        return f"invalid puzzle record ({self.reason}): {self.record!r}"


def normalize_word(word: str) -> str:
    return word.strip().upper()


def _parse_record(rec: dict) -> Puzzle:
    answers = rec.get("answers")
    if not isinstance(answers, list) or len(answers) != 4:
        raise PuzzleValidationError(rec, "expected 4 answer groups")
    groups = []
    for a in answers:
        members = a.get("members", [])
        if len(members) != 4:
            raise PuzzleValidationError(rec, "expected 4 members per group")
        groups.append(
            Group(
                level=int(a.get("level", 0)), name=str(a.get("group", "")), members=tuple(members)
            )
        )
    words = [normalize_word(w) for g in groups for w in g.members]
    if len(set(words)) != 16:
        raise PuzzleValidationError(rec, "expected 16 unique words")
    return Puzzle(
        puzzle_id=int(rec.get("puzzle_id", rec.get("id", -1))),
        date=str(rec.get("date", "")),
        groups=tuple(groups),
        strata=str(rec.get("category", DEFAULT_STRATUM)),
    )


def load_puzzles(path: str | Path, strict: bool = False) -> list[Puzzle]:
    """Load puzzles from a gvc-local-schema JSON file, sorted by date.

    Malformed records are skipped unless ``strict`` is True.
    """
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    puzzles: list[Puzzle] = []
    for rec in raw:
        try:
            puzzles.append(_parse_record(rec))
        except PuzzleValidationError:
            if strict:
                raise
    puzzles.sort(key=lambda p: (p.date, p.puzzle_id))
    return puzzles


def default_puzzles_path() -> Path:
    """Resolve the puzzle DB path.

    Order: $CONNECTIONS_PUZZLES, then a sibling gvc-local checkout
    (``../gvc-local/data/puzzles/tagged_connections.json``).
    """
    env = os.environ.get(PUZZLES_ENV_VAR)
    if env:
        return Path(env)
    here = Path(__file__).resolve()
    for parent in here.parents:
        candidate = parent.parent / "gvc-local" / "data" / "puzzles" / "tagged_connections.json"
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "Puzzle DB not found. Set $CONNECTIONS_PUZZLES to the path of "
        "gvc-local/data/puzzles/tagged_connections.json"
    )
