# connections-rl — make targets mirror the project milestones.
# All targets are idempotent.

PY ?= python
PUZZLES ?= $(CONNECTIONS_PUZZLES)

.PHONY: setup data train-sft train-grpo eval eval-smoke gate serve report lint test docker

setup:
	pip install -e ".[dev]"

## M0 — build leakage-aware splits + SFT chat data from the gvc-local puzzle DB.
data:
	$(PY) -m connections_rl.data.build --out data/splits

## M1 — LoRA SFT warm start (run on Colab T4 or any CUDA box).
train-sft:
	$(PY) -m connections_rl.train.sft --config configs/train/sft.yaml

## M1/M2 — GRPO from the SFT checkpoint. Use configs/accelerate/fsdp_2xt4.yaml on Kaggle.
train-grpo:
	$(PY) -m connections_rl.train.grpo --config configs/train/grpo.yaml

## M3 — full offline eval of every arm listed in configs/eval/default.yaml.
eval:
	$(PY) -m connections_rl.eval.run --config configs/eval/default.yaml

## CI gate — tiny deterministic eval on bundled fixtures; no network, no GPU.
eval-smoke:
	$(PY) -m connections_rl.eval.run --smoke

## Release gate — fail if candidate regresses vs baseline beyond the CI.
gate:
	$(PY) -m connections_rl.eval.gate \
		--candidate results/grpo/metrics.json --baseline results/sft/metrics.json

serve:
	uvicorn connections_rl.serve.app:create_app --factory --host 0.0.0.0 --port 8080

report:
	$(PY) -m connections_rl.report.build --results results --out report/results.md

lint:
	ruff check src tests
	ruff format --check src tests
	mypy src/connections_rl

test:
	pytest -v

docker:
	docker build -t connections-rl:latest .
