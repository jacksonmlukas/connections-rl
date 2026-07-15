"""Chat formatting: how a puzzle is shown to the model and what it must emit.

Single-pass protocol (contrast with gvc-local's multi-turn guess loop): the
model sees all 16 words once and must return all four groups in one shot,
inside an <ANSWER> block. Anything before the block (reasoning) is ignored by
the parser, which lets GRPO learn to use its own scratchpad.
"""

from __future__ import annotations

import random

from connections_rl.data.loader import Puzzle

SYSTEM_PROMPT = """\
You are an expert NYT Connections solver. You are given 16 words that form \
exactly 4 hidden groups of 4 words each. Every word belongs to exactly one group.

Think step by step if useful, then give your final answer in EXACTLY this format:

<ANSWER>
Group 1: WORD, WORD, WORD, WORD
Group 2: WORD, WORD, WORD, WORD
Group 3: WORD, WORD, WORD, WORD
Group 4: WORD, WORD, WORD, WORD
</ANSWER>

Rules: use each of the 16 given words exactly once, 4 words per group, \
spelled exactly as given. Do not add words. Do not repeat words."""


def shuffled_words(puzzle: Puzzle, seed: int | None = None) -> list[str]:
    """Deterministically shuffle the board so group order is not a giveaway."""
    words = list(puzzle.words)
    rng = random.Random(puzzle.puzzle_id if seed is None else seed)
    rng.shuffle(words)
    return words


def format_user(puzzle: Puzzle, seed: int | None = None) -> str:
    return "Words: " + ", ".join(shuffled_words(puzzle, seed))


def canonical_answer(puzzle: Puzzle) -> str:
    """The target completion for SFT: groups in difficulty order."""
    lines = [
        f"Group {i + 1}: " + ", ".join(g.members)
        for i, g in enumerate(sorted(puzzle.groups, key=lambda g: g.level))
    ]
    return "<ANSWER>\n" + "\n".join(lines) + "\n</ANSWER>"


def build_chat(puzzle: Puzzle, seed: int | None = None) -> list[dict[str, str]]:
    """Prompt-only chat (for inference / GRPO rollouts)."""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": format_user(puzzle, seed)},
    ]


def build_sft_example(puzzle: Puzzle, seed: int | None = None) -> dict:
    """OpenAI-chat-format SFT example (same shape as gvc-local's finetune data)."""
    return {
        "messages": build_chat(puzzle, seed)
        + [{"role": "assistant", "content": canonical_answer(puzzle)}],
        "puzzle_id": puzzle.puzzle_id,
        "date": puzzle.date,
        "strata": puzzle.strata,
    }
