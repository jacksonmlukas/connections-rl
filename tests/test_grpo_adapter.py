"""GRPO reward-adapter tests (no torch/TRL required)."""

import json

from connections_rl.data.build import puzzle_to_record
from connections_rl.data.formatting import canonical_answer
from connections_rl.data.loader import Puzzle
from connections_rl.reward.reward import RewardConfig
from connections_rl.train.grpo import make_reward_func


def test_reward_func_matches_direct_reward(puzzles: list[Puzzle]) -> None:
    fn = make_reward_func(RewardConfig())
    p = puzzles[0]
    rec = puzzle_to_record(p)
    board = [w.upper() for g in rec["answers"] for w in g["members"]]
    answer_sets = json.dumps([[w.upper() for w in g["members"]] for g in rec["answers"]])

    # Conversational completion shape (list of messages), as TRL passes it.
    good = [[{"role": "assistant", "content": canonical_answer(p)}]]
    bad = [[{"role": "assistant", "content": "no idea"}]]
    rewards = fn(
        prompts=[None, None],
        completions=good + bad,
        board=[board, board],
        answer_sets=[answer_sets, answer_sets],
    )
    cfg = RewardConfig()
    assert rewards[0] == cfg.max_reward
    assert rewards[1] == cfg.invalid_penalty


def test_reward_func_accepts_plain_strings(puzzles: list[Puzzle]) -> None:
    fn = make_reward_func(RewardConfig())
    p = puzzles[0]
    rec = puzzle_to_record(p)
    board = [w.upper() for g in rec["answers"] for w in g["members"]]
    answer_sets = json.dumps([[w.upper() for w in g["members"]] for g in rec["answers"]])
    rewards = fn(
        prompts=[None],
        completions=[canonical_answer(p)],
        board=[board],
        answer_sets=[answer_sets],
    )
    assert rewards[0] == RewardConfig().max_reward
