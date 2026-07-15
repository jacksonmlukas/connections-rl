"""Solver adapters for the eval harness.

``endpoint_solver`` targets any OpenAI-compatible server (vLLM, Groq,
Together) — the same interchangeability trick as gvc-local's EndpointConfig.
``oracle_solver`` / ``random_solver`` are deterministic arms used by unit
tests and the CI smoke eval.
"""

from __future__ import annotations

import os
import random
from collections.abc import Callable

from connections_rl.data.loader import Puzzle

Solver = Callable[[list[dict[str, str]]], str]

DEFAULT_BASE_URL = "http://localhost:8000/v1"


def endpoint_solver(
    model: str,
    base_url: str | None = None,
    api_key: str | None = None,
    temperature: float = 0.0,
    max_tokens: int = 1024,
) -> Solver:
    """Solver backed by an OpenAI-compatible /chat/completions endpoint."""
    from openai import OpenAI  # deferred: heavy optional dep

    client = OpenAI(
        base_url=base_url or os.environ.get("CRL_BASE_URL", DEFAULT_BASE_URL),
        api_key=api_key or os.environ.get("CRL_API_KEY", "EMPTY"),
    )

    def solve(messages: list[dict[str, str]]) -> str:
        resp = client.chat.completions.create(
            model=model,
            messages=messages,  # type: ignore[arg-type]
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return resp.choices[0].message.content or ""

    return solve


def _format_answer(groups: list[list[str]]) -> str:
    lines = [f"Group {i + 1}: " + ", ".join(g) for i, g in enumerate(groups)]
    return "<ANSWER>\n" + "\n".join(lines) + "\n</ANSWER>"


def oracle_solver(puzzles: list[Puzzle]) -> Solver:
    """Always answers correctly (upper-bound arm for smoke tests)."""
    by_words = {frozenset(p.words): p for p in puzzles}

    def solve(messages: list[dict[str, str]]) -> str:
        words = frozenset(
            w.strip() for w in messages[-1]["content"].removeprefix("Words: ").split(",")
        )
        p = by_words[words]
        return _format_answer([list(g.members) for g in sorted(p.groups, key=lambda g: g.level)])

    return solve


def random_solver(seed: int = 0) -> Solver:
    """Random valid partition of the board (chance-level arm)."""
    rng = random.Random(seed)

    def solve(messages: list[dict[str, str]]) -> str:
        words = [w.strip() for w in messages[-1]["content"].removeprefix("Words: ").split(",")]
        rng.shuffle(words)
        return _format_answer([words[i : i + 4] for i in range(0, 16, 4)])

    return solve
