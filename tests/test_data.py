"""Data pipeline tests: loading, round-trip, splits, formatting."""

import json
from pathlib import Path

import pytest

from connections_rl.data.build import puzzle_to_record
from connections_rl.data.formatting import (
    SYSTEM_PROMPT,
    build_chat,
    build_sft_example,
    canonical_answer,
    format_user,
    shuffled_words,
)
from connections_rl.data.loader import Puzzle, PuzzleValidationError, _parse_record, load_puzzles
from connections_rl.data.splits import split_by_date

FIXTURES = Path(__file__).parent / "fixtures"


def test_load_sample(puzzles: list[Puzzle]) -> None:
    assert len(puzzles) == 10
    for p in puzzles:
        assert len(p.groups) == 4
        assert len(p.words) == 16
        assert len(set(p.words)) == 16
        assert p.date and p.strata


def test_loader_sorts_by_date(puzzles: list[Puzzle]) -> None:
    dates = [p.date for p in puzzles]
    assert dates == sorted(dates)


def test_loader_skips_malformed_unless_strict(tmp_path: Path, puzzles: list[Puzzle]) -> None:
    good = puzzle_to_record(puzzles[0])
    bad = {"puzzle_id": 999, "date": "2024-01-01", "answers": [{"members": ["A"]}]}
    path = tmp_path / "mixed.json"
    path.write_text(json.dumps([good, bad]))
    assert len(load_puzzles(path)) == 1
    with pytest.raises(PuzzleValidationError):
        load_puzzles(path, strict=True)


def test_record_round_trip(puzzles: list[Puzzle]) -> None:
    for p in puzzles:
        assert _parse_record(puzzle_to_record(p)) == p


def test_split_by_date_no_leakage(puzzles: list[Puzzle]) -> None:
    split = split_by_date(puzzles, val_frac=0.2, test_frac=0.2)
    assert len(split.train) + len(split.val) + len(split.test) == len(puzzles)
    assert split.train and split.val and split.test
    assert max(p.date for p in split.train) < min(p.date for p in split.val)
    assert max(p.date for p in split.val) < min(p.date for p in split.test)


def test_split_deterministic(puzzles: list[Puzzle]) -> None:
    a = split_by_date(puzzles, 0.2, 0.2)
    b = split_by_date(puzzles, 0.2, 0.2)
    assert [p.puzzle_id for p in a.test] == [p.puzzle_id for p in b.test]


def test_split_rejects_bad_fractions(puzzles: list[Puzzle]) -> None:
    with pytest.raises(ValueError):
        split_by_date(puzzles, val_frac=0.6, test_frac=0.6)


def test_shuffle_is_deterministic_and_complete(puzzle: Puzzle) -> None:
    assert shuffled_words(puzzle) == shuffled_words(puzzle)
    assert sorted(shuffled_words(puzzle)) == sorted(puzzle.words)
    assert shuffled_words(puzzle, seed=1) != shuffled_words(puzzle, seed=2)


def test_chat_format(puzzle: Puzzle) -> None:
    chat = build_chat(puzzle)
    assert chat[0]["role"] == "system" and chat[0]["content"] == SYSTEM_PROMPT
    assert chat[1]["role"] == "user" and chat[1]["content"].startswith("Words: ")
    for w in puzzle.words:
        assert w in chat[1]["content"]


def test_canonical_answer_is_perfect_reward_input(puzzle: Puzzle) -> None:
    from connections_rl.reward.reward import parse_groups, validate_against_board

    groups = parse_groups(canonical_answer(puzzle))
    assert groups is not None
    assert validate_against_board(groups, puzzle.words)
    assert set(groups) == set(puzzle.answer_sets)


def test_sft_example_shape(puzzle: Puzzle) -> None:
    ex = build_sft_example(puzzle)
    roles = [m["role"] for m in ex["messages"]]
    assert roles == ["system", "user", "assistant"]
    assert ex["puzzle_id"] == puzzle.puzzle_id
    assert format_user(puzzle) == ex["messages"][1]["content"]


def test_build_cli_writes_splits(tmp_path: Path) -> None:
    from connections_rl.data.build import main

    out = tmp_path / "splits"
    main(
        [
            "--puzzles",
            str(FIXTURES / "puzzles_sample.json"),
            "--out",
            str(out),
            "--val-frac",
            "0.2",
            "--test-frac",
            "0.2",
        ]
    )
    for name in ("train", "val", "test"):
        assert (out / f"{name}.jsonl").exists()
        assert (out / f"puzzles_{name}.json").exists()
    summary = json.loads((out / "split_summary.json").read_text())
    assert summary["train"]["n"] + summary["val"]["n"] + summary["test"]["n"] == 10
