"""Serving API smoke tests with a stubbed model backend (no vLLM needed)."""

import pytest

fastapi = pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient  # noqa: E402

from connections_rl.data.formatting import canonical_answer  # noqa: E402
from connections_rl.data.loader import Puzzle  # noqa: E402
from connections_rl.serve.app import create_app  # noqa: E402


def make_client(puzzles: list[Puzzle], grpo_solves: bool = True) -> TestClient:
    by_words = {frozenset(p.words): p for p in puzzles}

    def stub_chat(
        model: str, messages: list[dict[str, str]], temperature: float, max_tokens: int
    ) -> str:
        words = frozenset(
            w.strip().upper() for w in messages[-1]["content"].removeprefix("Words: ").split(",")
        )
        p = by_words[words]
        if "grpo" in model.lower() and grpo_solves:
            return "I considered the board carefully.\n" + canonical_answer(p)
        return "Group 1: not a real answer"  # base model flubs the format

    return TestClient(create_app(chat_fn=stub_chat))


def test_health(puzzles: list[Puzzle]) -> None:
    client = make_client(puzzles)
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["uptime_s"] >= 0


def test_solve_grpo_valid(puzzles: list[Puzzle]) -> None:
    client = make_client(puzzles)
    p = puzzles[0]
    r = client.post("/solve", json={"words": p.words, "arm": "grpo"})
    assert r.status_code == 200
    body = r.json()
    assert body["valid"] is True
    assert len(body["groups"]) == 4
    assert {frozenset(g) for g in body["groups"]} == set(p.answer_sets)


def test_solve_base_invalid_completion_is_reported(puzzles: list[Puzzle]) -> None:
    client = make_client(puzzles)
    r = client.post("/solve", json={"words": puzzles[0].words, "arm": "base"})
    assert r.status_code == 200
    assert r.json()["valid"] is False
    assert r.json()["groups"] is None


def test_solve_rejects_bad_board(puzzles: list[Puzzle]) -> None:
    client = make_client(puzzles)
    r = client.post("/solve", json={"words": ["A", "B"], "arm": "grpo"})
    assert r.status_code == 422


def test_compare(puzzles: list[Puzzle]) -> None:
    client = make_client(puzzles)
    p = puzzles[1]
    r = client.post("/compare", json={"words": p.words})
    assert r.status_code == 200
    body = r.json()
    assert body["grpo"]["valid"] is True
    assert body["base"]["valid"] is False
    assert body["agree"] is False


def test_metrics_counts_requests(puzzles: list[Puzzle]) -> None:
    client = make_client(puzzles)
    client.post("/solve", json={"words": puzzles[0].words, "arm": "grpo"})
    client.post("/solve", json={"words": puzzles[0].words, "arm": "grpo"})
    m = client.get("/metrics").json()
    assert m["requests"]["/solve:grpo"] == 2
    assert m["latency_ms"]["n"] >= 2


def test_backend_error_is_502(puzzles: list[Puzzle]) -> None:
    def broken(model, messages, temperature, max_tokens):  # type: ignore[no-untyped-def]
        raise ConnectionError("vLLM offline")

    client = TestClient(create_app(chat_fn=broken))
    r = client.post("/solve", json={"words": puzzles[0].words, "arm": "grpo"})
    assert r.status_code == 502
