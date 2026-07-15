"""FastAPI serving layer: base vs. GRPO behind one API.

Run with:  uvicorn connections_rl.serve.app:create_app --factory

Environment:
  CRL_BASE_URL        OpenAI-compatible endpoint (vLLM), default http://localhost:8000/v1
  CRL_BASE_MODEL      served name of the base/SFT model
  CRL_GRPO_MODEL      served name of the GRPO model (vLLM can serve the LoRA adapter)
  CRL_API_KEY         bearer for the endpoint, default "EMPTY"

Endpoints: POST /solve, POST /compare, GET /health, GET /metrics.
The /compare endpoint is the demo money-shot: same board, base vs. RL, side
by side.
"""

from __future__ import annotations

import os
import time
from collections.abc import Callable

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from connections_rl.reward.reward import parse_groups, validate_against_board
from connections_rl.serve.models import (
    Arm,
    CompareResponse,
    HealthResponse,
    SolveRequest,
    SolveResponse,
)
from connections_rl.serve.monitoring import RequestMonitor

# Injectable chat function: (model, messages, temperature, max_tokens) -> completion.
ChatFn = Callable[[str, list[dict[str, str]], float, int], str]


def _default_chat_fn() -> ChatFn:
    from openai import OpenAI  # deferred: optional dep, and tests inject a stub

    client = OpenAI(
        base_url=os.environ.get("CRL_BASE_URL", "http://localhost:8000/v1"),
        api_key=os.environ.get("CRL_API_KEY", "EMPTY"),
    )

    def chat(
        model: str, messages: list[dict[str, str]], temperature: float, max_tokens: int
    ) -> str:
        resp = client.chat.completions.create(
            model=model,
            messages=messages,  # type: ignore[arg-type]
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return resp.choices[0].message.content or ""

    return chat


def create_app(chat_fn: ChatFn | None = None) -> FastAPI:
    app = FastAPI(title="connections-rl", version="0.1.0")
    app.add_middleware(
        CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
    )
    started = time.time()
    monitor = RequestMonitor()
    models = {
        Arm.base: os.environ.get("CRL_BASE_MODEL", "Qwen/Qwen2.5-1.5B-Instruct"),
        Arm.grpo: os.environ.get("CRL_GRPO_MODEL", "connections-rl-grpo"),
    }
    _chat: list[ChatFn | None] = [chat_fn]  # lazy so importing the app never needs openai

    def chat(
        model: str, messages: list[dict[str, str]], temperature: float, max_tokens: int
    ) -> str:
        fn = _chat[0]
        if fn is None:
            fn = _default_chat_fn()
            _chat[0] = fn
        return fn(model, messages, temperature, max_tokens)

    def solve_one(req: SolveRequest, arm: Arm) -> SolveResponse:
        from connections_rl.data.formatting import SYSTEM_PROMPT

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": "Words: " + ", ".join(req.words)},
        ]
        start = time.perf_counter()
        try:
            completion = chat(models[arm], messages, req.temperature, req.max_tokens)
        except Exception as exc:  # endpoint down, model missing, etc.
            monitor.record(f"/solve:{arm.value}", (time.perf_counter() - start) * 1000, ok=False)
            raise HTTPException(status_code=502, detail=f"model backend error: {exc}") from exc
        latency_ms = (time.perf_counter() - start) * 1000
        groups = parse_groups(completion)
        valid = groups is not None and validate_against_board(groups, req.words)
        monitor.record(f"/solve:{arm.value}", latency_ms, ok=True)
        return SolveResponse(
            arm=arm,
            model=models[arm],
            groups=[sorted(g) for g in groups] if groups and valid else None,
            valid=valid,
            raw_completion=completion,
            latency_ms=latency_ms,
        )

    @app.post("/solve", response_model=SolveResponse)
    def solve(req: SolveRequest) -> SolveResponse:
        return solve_one(req, req.arm)

    @app.post("/compare", response_model=CompareResponse)
    def compare(req: SolveRequest) -> CompareResponse:
        base = solve_one(req, Arm.base)
        grpo = solve_one(req, Arm.grpo)
        agree = bool(
            base.valid
            and grpo.valid
            and base.groups is not None
            and grpo.groups is not None
            and {frozenset(g) for g in base.groups} == {frozenset(g) for g in grpo.groups}
        )
        return CompareResponse(base=base, grpo=grpo, agree=agree)

    @app.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        return HealthResponse(
            status="ok",
            base_model=models[Arm.base],
            grpo_model=models[Arm.grpo],
            uptime_s=time.time() - started,
        )

    @app.get("/metrics")
    def metrics() -> dict:
        return monitor.summary()

    return app
