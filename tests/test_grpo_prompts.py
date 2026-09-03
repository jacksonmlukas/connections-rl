"""GRPO prompt construction must not leak answer-key order.

Regression test for the bug described in
results-analysis/aug27/GRPO_PROMPT_FIX.md: build_dataset formerly emitted the
sixteen words in answer-key order, so positions 1-4 were group 1, 5-8 group 2,
and so on. Every GRPO board was then solvable by a positional copy rule worth
the full reward, while SFT data and all evaluation used shuffled boards.
"""

import json
from pathlib import Path

from connections_rl.data.formatting import shuffled_words
from connections_rl.data.loader import load_puzzles
from connections_rl.train.grpo import build_dataset

FIXTURES = Path(__file__).parent / "fixtures"


def _records() -> list[dict]:
    return json.loads((FIXTURES / "puzzles_sample.json").read_text())


def _prompt_words(row) -> list[str]:
    user = next(m["content"] for m in row["prompt"] if m["role"] == "user")
    return [w.strip().upper() for w in user.replace("Words:", "", 1).split(",")]


def test_prompt_is_not_in_answer_key_order():
    recs = _records()
    ds = build_dataset(recs)
    for row, rec in zip(ds, recs, strict=True):
        answer_order = [w.upper() for g in rec["answers"] for w in g["members"]]
        got = _prompt_words(row)
        assert got != answer_order, f"puzzle {rec['puzzle_id']}: prompt leaks answer order"
        assert sorted(got) == sorted(answer_order), "shuffle must preserve the board"


def test_no_board_is_solvable_by_a_positional_copy():
    """The stronger property: consecutive quadruples must not be the answer key.

    A prompt could differ from answer order elementwise and still be copy-solvable
    if the shuffle only permuted within groups.
    """
    recs = _records()
    for row, rec in zip(build_dataset(recs), recs, strict=True):
        got = _prompt_words(row)
        quads = [set(got[i : i + 4]) for i in range(0, 16, 4)]
        key = [{w.upper() for w in g["members"]} for g in rec["answers"]]
        assert quads != key, f"puzzle {rec['puzzle_id']}: copy rule scores full reward"


def test_matches_the_canonical_shuffle_used_by_sft_and_eval():
    """One code path, one ordering: GRPO must agree with formatting.shuffled_words."""
    recs = _records()
    by_id = {p.puzzle_id: p for p in load_puzzles(FIXTURES / "puzzles_sample.json")}
    for row, rec in zip(build_dataset(recs), recs, strict=True):
        canonical = [w.upper() for w in shuffled_words(by_id[rec["puzzle_id"]])]
        assert _prompt_words(row) == canonical


def test_shuffle_is_deterministic_across_calls():
    recs = _records()
    a, b = build_dataset(recs), build_dataset(recs)
    assert [_prompt_words(r) for r in a] == [_prompt_words(r) for r in b]


def test_board_column_matches_the_prompt():
    """`board` feeds reward lookup; it must be the order the model actually saw."""
    for row in build_dataset(_records()):
        assert [w.upper() for w in row["board"]] == _prompt_words(row)


def test_answer_sets_are_unchanged_by_the_shuffle():
    recs = _records()
    for row, rec in zip(build_dataset(recs), recs, strict=True):
        got = [frozenset(g) for g in json.loads(row["answer_sets"])]
        want = [frozenset(w.upper() for w in g["members"]) for g in rec["answers"]]
        assert sorted(map(sorted, got)) == sorted(map(sorted, want))
