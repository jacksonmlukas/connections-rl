"""Reward function tests: the RL core must be watertight."""

import pytest

from connections_rl.data.formatting import canonical_answer
from connections_rl.data.loader import Puzzle
from connections_rl.reward.reward import (
    RewardConfig,
    parse_groups,
    reward,
    reward_breakdown,
    score_groups,
    validate_against_board,
)

CFG = RewardConfig()


def perfect(puzzle: Puzzle) -> str:
    return canonical_answer(puzzle)


# ---------- parsing ----------


def test_parse_canonical_answer(puzzle: Puzzle) -> None:
    groups = parse_groups(perfect(puzzle))
    assert groups is not None and len(groups) == 4
    assert validate_against_board(groups, puzzle.words)


def test_parse_ignores_reasoning_before_answer(puzzle: Puzzle) -> None:
    text = "Let me think... Group 1: A, B, C, D looks tempting but no.\n" + perfect(puzzle)
    groups = parse_groups(text)
    assert groups is not None
    assert validate_against_board(groups, puzzle.words)


def test_parse_uses_last_answer_block(puzzle: Puzzle) -> None:
    wrong = perfect(puzzle).replace(puzzle.words[0], "NOTAWORD", 1)
    text = wrong + "\nWait, I made a mistake.\n" + perfect(puzzle)
    groups = parse_groups(text)
    assert groups is not None
    assert validate_against_board(groups, puzzle.words)


def test_parse_without_answer_tags(puzzle: Puzzle) -> None:
    text = perfect(puzzle).replace("<ANSWER>", "").replace("</ANSWER>", "")
    assert parse_groups(text) is not None


def test_parse_case_insensitive_and_whitespace(puzzle: Puzzle) -> None:
    text = perfect(puzzle).lower().replace("group", "  gRoUp ")
    groups = parse_groups(text)
    assert groups is not None
    assert validate_against_board(groups, puzzle.words)


@pytest.mark.parametrize(
    "bad",
    [
        "",
        "no groups here",
        "<ANSWER>\nGroup 1: A, B, C\n</ANSWER>",  # 3 words
        "<ANSWER>\nGroup 1: A, B, C, D\nGroup 2: E, F, G, H\n</ANSWER>",  # 2 groups
        "<ANSWER>\nGroup 1: A, A, B, C\nGroup 2: D, E, F, G\nGroup 3: H, I, J, K\nGroup 4: L, M, N, O\n</ANSWER>",  # dup in group
    ],
)
def test_parse_rejects_malformed(bad: str) -> None:
    assert parse_groups(bad) is None


def test_board_validation_rejects_wrong_words(puzzle: Puzzle) -> None:
    text = perfect(puzzle).replace(puzzle.words[0], "ZZZZZ", 1)
    groups = parse_groups(text)
    assert groups is not None  # parses fine...
    assert not validate_against_board(groups, puzzle.words)  # ...but fails the board check
    assert reward(text, puzzle.words, puzzle.answer_sets) == CFG.invalid_penalty


def test_board_validation_rejects_reused_word(puzzle: Puzzle) -> None:
    words = puzzle.words
    text = perfect(puzzle).replace(words[4], words[0], 1)  # word 0 appears twice
    groups = parse_groups(text)
    if groups is not None:
        assert not validate_against_board(groups, words)


# ---------- scoring ----------


def test_score_all_correct(puzzle: Puzzle) -> None:
    correct, one_away = score_groups(puzzle.answer_sets, puzzle.answer_sets)
    assert (correct, one_away) == (4, 0)


def test_score_two_swapped_words(puzzle: Puzzle) -> None:
    a, b, c, d = (set(s) for s in puzzle.answer_sets)
    w_a, w_b = next(iter(a)), next(iter(b))
    a2 = frozenset(a - {w_a} | {w_b})
    b2 = frozenset(b - {w_b} | {w_a})
    pred = [a2, b2, frozenset(c), frozenset(d)]
    correct, one_away = score_groups(pred, puzzle.answer_sets)
    assert correct == 2
    assert one_away == 2  # each swapped group is one word away


# ---------- reward totals ----------


def test_reward_perfect(puzzle: Puzzle) -> None:
    br = reward_breakdown(perfect(puzzle), puzzle.words, puzzle.answer_sets)
    assert br.valid and br.solved and br.correct_groups == 4
    assert br.total == pytest.approx(CFG.format_reward + CFG.grouping_weight + CFG.solve_bonus)
    assert br.total == pytest.approx(CFG.max_reward)


def test_reward_invalid_is_penalized(puzzle: Puzzle) -> None:
    br = reward_breakdown("gibberish", puzzle.words, puzzle.answer_sets)
    assert not br.valid and br.total == CFG.invalid_penalty
    assert br.total < 0  # strictly worse than any valid attempt


def test_reward_monotone_in_correct_groups(puzzle: Puzzle) -> None:
    """More correct groups must never earn less reward (anti-reward-hacking)."""
    a, b, c, d = (list(s) for s in puzzle.answer_sets)
    # 0 correct: rotate one word between every pair of groups
    scramble = [
        [a[0], a[1], a[2], b[0]],
        [b[1], b[2], b[3], c[0]],
        [c[1], c[2], c[3], d[0]],
        [d[1], d[2], d[3], a[3]],
    ]

    def as_text(groups: list[list[str]]) -> str:
        lines = [f"Group {i + 1}: " + ", ".join(g) for i, g in enumerate(groups)]
        return "<ANSWER>\n" + "\n".join(lines) + "\n</ANSWER>"

    r0 = reward(as_text(scramble), puzzle.words, puzzle.answer_sets)
    two_swapped = [
        [a[0], a[1], a[2], b[3]],
        [b[0], b[1], b[2], a[3]],
        c,
        d,
    ]
    r2 = reward(as_text(two_swapped), puzzle.words, puzzle.answer_sets)
    r4 = reward(as_text([a, b, c, d]), puzzle.words, puzzle.answer_sets)
    assert CFG.invalid_penalty < r0 < r2 < r4


def test_reward_config_overrides(puzzle: Puzzle) -> None:
    cfg = RewardConfig(solve_bonus=2.0, format_reward=0.0, one_away_credit=0.0)
    assert reward(perfect(puzzle), puzzle.words, puzzle.answer_sets, cfg) == pytest.approx(3.0)
