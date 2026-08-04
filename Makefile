# connections-rl — make targets mirror the project milestones.
# All targets are idempotent.

PY ?= python
PUZZLES ?= $(CONNECTIONS_PUZZLES)

.PHONY: setup data train-sft train-grpo eval eval-smoke gate serve report cards cards-check push-cards lint test docker

setup:
	pip install -e ".[dev]"

## M0 — build leakage-aware splits + SFT chat data from the gvc-local puzzle DB.
data:
	$(PY) -m connections_rl.data.build --out data/splits

## M1 — LoRA SFT warm start (run on Colab T4 or any CUDA box).
train-sft:
	$(PY) -m connections_rl.train.sft --config configs/train/sft.yaml

## M1/M2 — GRPO from the SFT checkpoint. Single-GPU: every published run is single-T4.
## (configs/accelerate/fsdp_2xt4.yaml is kept as a documented pre-Ampere failure, not a path to use.)
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

## Model cards — regenerate hub_cards/*.md from the committed eval JSON.
cards:
	$(PY) scripts/build_model_cards.py

## CI check — fail if any card disagrees with the results it cites.
cards-check:
	$(PY) scripts/build_model_cards.py --check

## Publish cards to the Hub (needs HF_TOKEN with write scope).
push-cards:
	$(PY) scripts/push_model_cards.py

lint:
	ruff check src tests
	ruff format --check src tests
	mypy src/connections_rl

test:
	pytest -v

docker:
	docker build -t connections-rl:latest .
